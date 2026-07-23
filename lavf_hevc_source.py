# -*- coding: utf-8 -*-
"""Source HEVC = demux (avformat) + decode (avcodec) combines, en ctypes sur les
DLLs ffmpeg 8 bundlees. Spec: docs/superpowers/specs/2026-07-21-hevc-avcodec-d3d11va-design.md
Contrairement au chemin H.264 (lavf -> NALs Annex-B -> edge264), avformat alimente
avcodec DIRECTEMENT. Perimetre v1: HEVC 4:2:0 8/10-bit; tout le reste -> open()=None."""
import os
import ctypes
import logging
from dataclasses import dataclass

import numpy as np

import lavf_h264_demuxer as _lavf   # reutilise _load() + structures + offsets (DRY)
from lavf_h264_demuxer import AVRational, AVFormatContext, AVCodecParameters, AVPacket

logger = logging.getLogger(__name__)

_AVMEDIA_TYPE_VIDEO = 0
_AV_NOPTS = -9223372036854775808
_AVSEEK_FLAG_BACKWARD = 1
_AVERROR_EAGAIN = -11            # AVERROR(EAGAIN), errno Windows/MinGW = 11
_AVERROR_EOF = -541478725        # FFERRTAG('E','O','F',' ')
_AV_OPT_SEARCH_CHILDREN = 1      # av_opt_get/set flag: options privees sur priv_data (MV-1 verbatim)


class AVFrame(ctypes.Structure):
    # Champs de TETE d'AVFrame, ABI-stables (ffmpeg 8 / avutil 60): data, linesize,
    # extended_data, width, height, nb_samples, format, pict_type, SAR, pts.
    # pts est VERIFIE au runtime (les pts recus doivent appartenir aux pts envoyes).
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8) * 8),
        ("linesize", ctypes.c_int * 8),
        ("extended_data", ctypes.c_void_p),
        ("width", ctypes.c_int), ("height", ctypes.c_int),
        ("nb_samples", ctypes.c_int), ("format", ctypes.c_int),
        ("pict_type", ctypes.c_int),
        ("sample_aspect_ratio", AVRational),
        ("pts", ctypes.c_int64),
    ]


_SIGNED = False
_SIGNED_HW = False
_PIX = {}          # nom -> valeur d'enum, resolus au runtime (ABI-proof)
# Fix-3 (MV-5 final review): sentinel for av_stereo3d_primary_eye_name availability.
# This symbol is an INFORMATIONAL cross-check only (_probe_multiview logging), never
# the left/right decision path (_map_left_view_id owns that) — a DLL build missing it
# must not break _sign()/is_available() (which would disable the entire HEVC path).
# None until _sign() resolves it; the (already-tolerant) use site checks this sentinel
# before calling the symbol.
_HAS_STEREO3D_EYE_NAME = None

# Offsets des DEUX seuls champs d'AVCodecContext ecrits par offset dans le chemin
# d'init HW (get_format, hw_device_ctx). SONDES (jamais comptes a la main) par
# tools/probe_avcodec_offsets.c compile contre les headers ffmpeg-8.0 avec
# C:\msys64\mingw64\bin\gcc.exe. VALABLES UNIQUEMENT pour LIBAVCODEC major 62.
# Sortie verbatim du probe : OFF_GET_FORMAT 192 / OFF_HW_DEVICE_CTX 560 / AVCODEC_MAJOR 62.
# Contre-verifies au runtime DANS le bloc d'arm HW d'open() (M4) : sur un AVCodecContext
# hevc frais, get_format @192 doit etre non-NULL (default installe par avcodec) et
# hw_device_ctx @560 doit etre NULL ; si l'une des deux attentes echoue, l'arm HW est saute
# (SW continue) plutot que de corrompre le contexte.
_OFF_GET_FORMAT = 192
_OFF_HW_DEVICE_CTX = 560

# Color-metadata field offsets of AVCodecParameters (HDR10 PQ/BT.2020 plumbing). SONDES
# par tools/probe_avcodec_offsets.c (meme recette que ci-dessus : gcc msys64 + headers
# ffmpeg-8.0). Sortie verbatim du probe :
#   OFF_COLOR_RANGE 100 / OFF_COLOR_PRIMARIES 104 / OFF_COLOR_TRC 108 / OFF_COLOR_SPACE 112
# VALABLES POUR LIBAVCODEC major 62 UNIQUEMENT (meme garde que get_format/hw_device_ctx).
# Contre-verifies au runtime PAR NOM (av_color_space_name / av_color_transfer_name) dans
# _read_color_metadata : sur un nom implausible, repli sur l'heuristique (largeur).
_OFF_COLOR_RANGE = 100
_OFF_COLOR_PRIMARIES = 104
_OFF_COLOR_TRC = 108
_OFF_COLOR_SPACE = 112

# Tokens plausibles renvoyes par av_color_*_name (avutil 60) — sert de garde ABI : si le nom
# lu a l'offset pinne n'en fait pas partie, l'offset ne tombe pas ou on croit -> heuristique.
_PLAUSIBLE_CS = frozenset((
    b'rgb', b'bt709', b'unspecified', b'reserved', b'fcc', b'bt470bg', b'smpte170m',
    b'smpte240m', b'ycgco', b'bt2020nc', b'bt2020c', b'smpte2085', b'chroma-derived-nc',
    b'chroma-derived-c', b'ictcp', b'ipt-c2', b'ycgco-re', b'ycgco-ro', b'unknown'))
_PLAUSIBLE_TRC = frozenset((
    b'reserved', b'bt709', b'unspecified', b'gamma22', b'gamma28', b'smpte170m',
    b'smpte240m', b'linear', b'log100', b'log316', b'iec61966-2-4', b'bt1361e',
    b'iec61966-2-1', b'bt2020-10', b'bt2020-20', b'smpte2084', b'smpte428',
    b'arib-std-b67', b'unknown'))

# get_format(AVCodecContext*, const enum AVPixelFormat* fmts) -> enum AVPixelFormat
_GET_FORMAT_CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p,
                                  ctypes.POINTER(ctypes.c_int))


def _sign():
    """Signe les fonctions avcodec/avutil dont ce module a besoin (une fois)."""
    global _SIGNED
    _lavf._load()
    if _SIGNED:
        return
    ac, au = _lavf._AVCODEC, _lavf._AVUTIL
    ac.avcodec_version.restype = ctypes.c_uint
    ac.avcodec_get_name.argtypes = [ctypes.c_int]
    ac.avcodec_get_name.restype = ctypes.c_char_p
    ac.avcodec_find_decoder_by_name.argtypes = [ctypes.c_char_p]
    ac.avcodec_find_decoder_by_name.restype = ctypes.c_void_p
    ac.avcodec_alloc_context3.argtypes = [ctypes.c_void_p]
    ac.avcodec_alloc_context3.restype = ctypes.c_void_p
    ac.avcodec_free_context.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    ac.avcodec_parameters_to_context.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    ac.avcodec_parameters_to_context.restype = ctypes.c_int
    ac.avcodec_open2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    ac.avcodec_open2.restype = ctypes.c_int
    ac.avcodec_send_packet.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    ac.avcodec_send_packet.restype = ctypes.c_int
    ac.avcodec_receive_frame.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    ac.avcodec_receive_frame.restype = ctypes.c_int
    ac.avcodec_flush_buffers.argtypes = [ctypes.c_void_p]
    au.av_opt_set.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    au.av_opt_set.restype = ctypes.c_int
    # MV-2 (multiview): av_opt_get lit des options string malloc'ees (ex "view_ids_available"),
    # a liberer via av_freep (contrat verifie MV-1: opt_get_str/opt_set_str du probe).
    au.av_opt_get.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                              ctypes.POINTER(ctypes.c_void_p)]
    au.av_opt_get.restype = ctypes.c_int
    au.av_freep.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    au.av_freep.restype = None
    au.av_frame_alloc.restype = ctypes.c_void_p
    au.av_frame_free.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    au.av_frame_unref.argtypes = [ctypes.c_void_p]
    au.av_get_pix_fmt.argtypes = [ctypes.c_char_p]
    au.av_get_pix_fmt.restype = ctypes.c_int
    au.av_frame_get_side_data.argtypes = [ctypes.c_void_p, ctypes.c_int]
    au.av_frame_get_side_data.restype = ctypes.c_void_p
    au.av_frame_side_data_name.argtypes = [ctypes.c_int]
    au.av_frame_side_data_name.restype = ctypes.c_char_p
    # Color-enum -> NAME resolvers (HDR10 plumbing). Used to cross-check the pinned
    # AVCodecParameters color offsets at runtime AND to expose readable names.
    au.av_color_space_name.argtypes = [ctypes.c_int]
    au.av_color_space_name.restype = ctypes.c_char_p
    au.av_color_transfer_name.argtypes = [ctypes.c_int]
    au.av_color_transfer_name.restype = ctypes.c_char_p
    au.av_color_primaries_name.argtypes = [ctypes.c_int]
    au.av_color_primaries_name.restype = ctypes.c_char_p
    # MV-2: resolveur NOM de l'oeil primaire AVStereo3D (cross-check informatif du
    # mapping left_view_id, JAMAIS le decideur — cf. _probe_multiview/_map_left_view_id).
    # Fix-3 (MV-5 final review): isolated in its own try/except — this symbol is only
    # ever used for an informational log cross-check (never the decision path), so a
    # DLL/build missing it must NOT break _sign() (and therefore is_available(), which
    # would silently disable the ENTIRE HEVC path over one optional symbol).
    global _HAS_STEREO3D_EYE_NAME
    try:
        au.av_stereo3d_primary_eye_name.argtypes = [ctypes.c_uint]
        au.av_stereo3d_primary_eye_name.restype = ctypes.c_char_p
        _HAS_STEREO3D_EYE_NAME = True
    except Exception as e:
        logger.warning(f"[HEVC] av_stereo3d_primary_eye_name indisponible "
                        f"(cross-check primary_eye desactive): {e}")
        _HAS_STEREO3D_EYE_NAME = False
    for name in ("yuv420p", "yuv420p10le", "nv12", "p010le", "d3d11"):
        _PIX[name] = au.av_get_pix_fmt(name.encode())
    _SIGNED = True


def _sign_hw():
    """Signe les fonctions avutil du chemin HW D3D11VA (une fois). Distinct de
    _sign() : n'est appele QUE quand allow_hw arme l'init HW, donc un avutil
    depourvu d'un symbole ne casse jamais le chemin SW."""
    global _SIGNED_HW
    if _SIGNED_HW:
        return
    _lavf._load()
    au = _lavf._AVUTIL
    au.av_hwdevice_find_type_by_name.argtypes = [ctypes.c_char_p]
    au.av_hwdevice_find_type_by_name.restype = ctypes.c_int
    au.av_hwdevice_ctx_create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                                          ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int]
    au.av_hwdevice_ctx_create.restype = ctypes.c_int
    au.av_buffer_ref.argtypes = [ctypes.c_void_p]
    au.av_buffer_ref.restype = ctypes.c_void_p
    au.av_buffer_unref.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    au.av_hwframe_transfer_data.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    au.av_hwframe_transfer_data.restype = ctypes.c_int
    _SIGNED_HW = True


def is_available():
    try:
        _sign()
        return True
    except Exception as e:
        logger.warning(f"[HEVC] ffmpeg DLLs indisponibles: {e}")
        return False


def _decode_one_probe_frame(ctx, vidx, dec, pkt, frame):
    """Decode UNE frame dans un contexte JETABLE (sonde multiview MV-2, cf.
    LavfHevcSource._probe_multiview) -> True si reussi, False sinon (EOF/erreur).
    Best-effort STRICT: ne leve JAMAIS (la sonde ne doit jamais faire echouer un
    open() qui aurait pu reussir en mono-vue)."""
    AF, AC = _lavf._AVFORMAT, _lavf._AVCODEC
    pkt_view = ctypes.cast(pkt, ctypes.POINTER(AVPacket))
    try:
        while True:
            r = AC.avcodec_receive_frame(dec, frame)
            if r == 0:
                return True
            if r == _AVERROR_EOF:
                return False
            if r != _AVERROR_EAGAIN:
                return False
            if AF.av_read_frame(ctx, pkt) < 0:
                AC.avcodec_send_packet(dec, None)
                continue
            pk = pkt_view.contents
            if pk.stream_index != vidx or pk.size <= 0:
                AC.av_packet_unref(pkt)
                continue
            s = AC.avcodec_send_packet(dec, pkt)
            AC.av_packet_unref(pkt)
            if s < 0 and s != _AVERROR_EAGAIN:
                return False
    except Exception:
        return False


@dataclass
class MediaInfo:
    width: int
    height: int
    bit_depth: int
    pix_fmt_name: str
    duration_ms: int          # 0 si inconnu (le player a sa propre duree via mpv)
    stereo_hint: str | None   # 'sbs' | 'tab' | None (side-data conteneur/SEI)
    stereo_inverted: bool
    # NOMS de color-space/transfer resolus (jamais des ints). Ex: 'bt2020nc'/'smpte2084'
    # (lus au runtime + contre-verifies par nom), ou heuristique '709'/'601' / None.
    color_space: str | None = None
    color_trc: str | None = None
    # MV-HEVC (Task MV-2): multiview (2 vues, spec §3). left_view_id = id de la vue
    # (0 ou 1, PAS un index de position) mappee sur l'oeil gauche — cf. LavfHevcSource._probe_multiview.
    multiview: bool = False
    left_view_id: int | None = None


class LavfHevcSource:
    def __init__(self):
        self._ctx = ctypes.c_void_p()
        self._dec = None            # AVCodecContext*
        self._pkt = None
        self._frame = None
        self._swframe = None        # AVFrame* SW temp pour le copy-back HW (Task 10)
        self._vidx = -1
        self._tb_num, self._tb_den = 1, 1000
        self._opened = False
        self._sent_pts = set()      # verif runtime de l'offset pts d'AVFrame
        self._pts_verified = False
        self._err_streak = 0
        self.failed = False
        self.info = None
        # --- etat D3D11VA (Task 9) ---
        self._hw_wanted = False      # get_format+hw_device_ctx poses avant open2
        self._hw_on = False          # get_format A REELLEMENT retenu d3d11
        self._hw_ref = None          # AVBufferRef* du device (notre reference)
        self._keep_cb = None         # ref DURE sur le CFUNCTYPE (anti-GC = crash)
        # --- etat multiview (Task MV-2) ---
        self._mv_pending = None      # snapshot (dict) de la vue en attente de sa partenaire pts
        # Fix-4 (MV-5 final review): les VRAIS ids probes (ex b"0,1"), exposes pour que
        # le player logue les MEMES ids que ceux reellement utilises au re-open (au lieu
        # d'un literal fige) ; None hors multiview / avant probe.
        self._mv_view_ids = None

    def open(self, path, allow_hw=False, _view_ids=None, _no_views=False,
             _left_view_id=None, _primary_eye=None):
        """_view_ids/_no_views/_left_view_id/_primary_eye: kwargs INTERNES (jamais
        destines a l'appelant), meme patron que le retry SW silencieux HW existant
        (self.open(path, allow_hw=False)).
        _view_ids=b"0,1": ceci EST la tentative de re-open multiview (spec §3) — pose
        av_opt_set(view_ids) AVANT open2, force allow_hw=False, saute la (re)detection
        (deja tranchee par _probe_multiview, cf. plus bas) ; tout echec dans cette
        branche -> repli silencieux mono-vue (jamais un refus, cf. except plus bas).
        _left_view_id/_primary_eye: mapping oeil precalcule par _probe_multiview (voir
        pourquoi ce n'est PAS relu ici dans le docstring de _probe_multiview : sous
        frame-threading, forced pour la perf, les options view_ids_available/
        view_pos_available ne se peuplent JAMAIS, meme apres decode).
        _no_views=True: repli mono-vue final (apres un echec de re-open multiview) —
        saute AUSSI la (re)detection, pour ne jamais reboucler sur une tentative
        multiview qui a deja echoue."""
        try:
            _sign()
        except Exception as e:
            logger.error(f"[HEVC] load failed: {e}")
            return None
        if self._opened:
            return self.info
        # --- MV-2 : detection multiview (spec §3), AVANT tout le reste. Sonde JETABLE
        # (contexte separe, cf. _probe_multiview) — seulement sur l'open "normal" (ni
        # re-open multiview deja en cours, ni repli mono-vue final), sinon on
        # rebouclerait indefiniment sur une tentative deja tranchee.
        if _view_ids is None and not _no_views:
            ids, mv_left_id, mv_eye = self._probe_multiview(path)
            if len(ids) >= 2:
                # Fix-4 (MV-5 final review): use the REAL probed ids (was a hardcoded
                # b"0,1" literal) so a file whose view_ids_available isn't literally
                # "0,1" (e.g. a different id pair) re-opens with the ids that actually
                # exist in this stream's VPS, instead of ids that may not.
                mv_view_ids = b','.join(ids[:2])
                self._mv_view_ids = mv_view_ids  # Fix-4: exposed for the player's log
                logger.info(f"[HEVC] multiview detecte (view_ids_available="
                            f"{b','.join(ids)!r}) -> re-open avec view_ids")
                return self.open(path, allow_hw=allow_hw, _view_ids=mv_view_ids,
                                 _left_view_id=mv_left_id, _primary_eye=mv_eye)
        AF, AC, AU = _lavf._AVFORMAT, _lavf._AVCODEC, _lavf._AVUTIL
        if AF.avformat_open_input(ctypes.byref(self._ctx),
                                  str(path).encode('utf-8'), None, None) < 0:
            return None
        try:
            if AF.avformat_find_stream_info(self._ctx, None) < 0:
                raise RuntimeError("find_stream_info")
            self._vidx = AF.av_find_best_stream(self._ctx, _AVMEDIA_TYPE_VIDEO, -1, -1, None, 0)
            if self._vidx < 0:
                raise RuntimeError("no video stream")
            fmt = ctypes.cast(self._ctx, ctypes.POINTER(AVFormatContext))
            sptr = fmt.contents.streams[self._vidx]
            cp_addr = ctypes.c_void_p.from_address(sptr + _lavf._OFF_CODECPAR).value
            if not cp_addr:
                raise RuntimeError("codecpar NULL")
            cp = AVCodecParameters.from_address(cp_addr)
            # codec_id lu par offset, CONTRE-VERIFIE par nom (ABI-proof)
            if (cp.codec_type != _AVMEDIA_TYPE_VIDEO
                    or AC.avcodec_get_name(cp.codec_id) != b"hevc"):
                logger.info(f"[HEVC] pas du HEVC (id={cp.codec_id}) -> refus")
                raise RuntimeError("not hevc")
            tb = AVRational.from_address(sptr + _lavf._OFF_TIME_BASE)
            if 0 < tb.den <= 1_000_000_000 and 0 < tb.num <= 1_000_000_000:
                self._tb_num, self._tb_den = tb.num, tb.den
            codec = AC.avcodec_find_decoder_by_name(b"hevc")
            if not codec:
                raise RuntimeError("decoder hevc absent")
            self._dec = AC.avcodec_alloc_context3(codec)
            if not self._dec:
                raise RuntimeError("alloc failed: avcodec_alloc_context3")
            if AC.avcodec_parameters_to_context(self._dec, cp_addr) < 0:
                raise RuntimeError("parameters_to_context")
            AU.av_opt_set(self._dec, b"threads", b"0", 0)   # frame-threading auto
            # --- MV-2 : view_ids AVANT open2 (spec §3). allow_hw force False en
            # multiview (le hwaccel D3D11VA ne couvre pas la vue dependante) ---
            effective_allow_hw = allow_hw and _view_ids is None
            if _view_ids is not None:
                r_vid = AU.av_opt_set(self._dec, b"view_ids", _view_ids, _AV_OPT_SEARCH_CHILDREN)
                if r_vid != 0:
                    raise RuntimeError(f"av_opt_set(view_ids={_view_ids!r}) err {r_vid}")
            # --- Init D3D11VA (Task 9) : get_format + hw_device_ctx AVANT open2 ---
            # Chaine de gardes : opt-in ET pas de kill-switch ET ABI major 62 ET
            # offsets sondes. TOUT echec ici laisse le SW intact (HW desactive en
            # silence : _hw_wanted reste False -> get_format jamais pose).
            if (effective_allow_hw
                    and os.environ.get('SYLC_HEVC_HW') != '0'
                    and (AC.avcodec_version() >> 16) == 62
                    and _OFF_GET_FORMAT is not None
                    and _OFF_HW_DEVICE_CTX is not None):
                try:
                    _sign_hw()
                    # M4: cheap runtime cross-check of the pinned offsets BEFORE writing to
                    # them (the offsets comment above promised this, but it was never
                    # implemented). A freshly alloc'd AVCodecContext has a non-NULL default
                    # get_format and a NULL hw_device_ctx; if either expectation fails the
                    # offsets don't land where we think -> skip HW arming (SW continues)
                    # instead of corrupting the context by writing at the wrong address.
                    _gf = ctypes.c_void_p.from_address(self._dec + _OFF_GET_FORMAT).value
                    _hwctx = ctypes.c_void_p.from_address(self._dec + _OFF_HW_DEVICE_CTX).value
                    if not _gf or _hwctx:
                        logger.warning(
                            f"[HEVC] cross-check offsets AVCodecContext echoue "
                            f"(get_format={_gf!r} attendu non-NULL, "
                            f"hw_device_ctx={_hwctx!r} attendu NULL) -> HW desarme, SW")
                        raise RuntimeError("AVCodecContext offset cross-check failed")
                    self._hw_ref = ctypes.c_void_p()
                    hwtype = AU.av_hwdevice_find_type_by_name(b"d3d11va")
                    if hwtype > 0 and AU.av_hwdevice_ctx_create(
                            ctypes.byref(self._hw_ref), hwtype, b"0", None, 0) == 0:
                        # hw_device_ctx = av_buffer_ref(ref) : le ctx consomme SA
                        # reference (liberee par avcodec_free_context) ; on garde la
                        # notre (self._hw_ref) et on l'unref dans close().
                        ctypes.c_void_p.from_address(self._dec + _OFF_HW_DEVICE_CTX).value = \
                            AU.av_buffer_ref(self._hw_ref)
                        self._keep_cb = _GET_FORMAT_CB(self._on_get_format)  # anti-GC!
                        ctypes.c_void_p.from_address(self._dec + _OFF_GET_FORMAT).value = \
                            ctypes.cast(self._keep_cb, ctypes.c_void_p).value
                        self._hw_wanted = True
                        logger.info("[HEVC] D3D11VA arme (get_format+hw_device_ctx poses)")
                    else:
                        logger.info("[HEVC] d3d11va indisponible -> SW")
                        self._hw_ref = None
                except Exception as e:
                    logger.info(f"[HEVC] init HW echouee -> SW: {e}")
                    # Fuite de ref device si l'exception survient APRES un
                    # av_hwdevice_ctx_create reussi (self._hw_ref alloue) : on
                    # l'unref AVANT de le nuller (revue Task 9, finding #2).
                    hw = self._hw_ref
                    if hw is not None:
                        ref = hw if isinstance(hw, ctypes.c_void_p) else ctypes.c_void_p(hw)
                        if ref.value and getattr(AU, 'av_buffer_unref', None):
                            AU.av_buffer_unref(ctypes.byref(ref))
                    self._hw_wanted, self._hw_on, self._hw_ref = False, False, None
            if AC.avcodec_open2(self._dec, codec, None) < 0:
                raise RuntimeError("avcodec_open2")
            self._pkt = AC.av_packet_alloc()
            if not self._pkt:
                raise RuntimeError("alloc failed: av_packet_alloc")
            self._frame = AU.av_frame_alloc()
            if not self._frame:
                raise RuntimeError("alloc failed: av_frame_alloc")
            self._opened = True
            # Decode la 1re frame pour connaitre pix_fmt/bit depth REELS + verifier pts
            first = self._decode_next()
            if first is None:
                raise RuntimeError("aucune frame decodable")
            fr = ctypes.cast(self._frame, ctypes.POINTER(AVFrame)).contents
            if not (0 < fr.width <= 16384 and 0 < fr.height <= 16384):
                raise RuntimeError("AVFrame width/height implausible (ABI)")
            fmt_i = fr.format
            if self._hw_on and fmt_i == _PIX["d3d11"]:
                # Frame HW (surface D3D11) : la profondeur reelle se lit via un
                # transfert one-shot vers une frame SW (nv12->8, p010le->10). Le
                # download plein (read_frame) est Task 10, qui reutilise ce chemin.
                bd = self._probe_hw_bit_depth()
                if bd is None:
                    # Echec du transfer-probe HW: retry SW silencieux (contrat
                    # "tout echec HW -> SW"), jamais de refus pour un caprice HW.
                    logger.warning("[HEVC] transfer-probe HW en echec -> retry SW")
                    self.close()
                    return self.open(path, allow_hw=False)
                name = "d3d11"
            elif fmt_i == _PIX["yuv420p"]:
                bd, name = 8, "yuv420p"
            elif fmt_i == _PIX["yuv420p10le"]:
                bd, name = 10, "yuv420p10le"
            else:
                logger.info(f"[HEVC] pix_fmt {fmt_i} hors perimetre -> refus (mpv)")
                raise RuntimeError("pix_fmt hors perimetre")
            hint, inv = self._read_stereo_side_data(fr)
            cs_name, trc_name = self._read_color_metadata(cp_addr, fr.width)
            multiview = _view_ids is not None
            left_view_id = _left_view_id if multiview else None
            self.info = MediaInfo(fr.width, fr.height, bd, name, 0, hint, inv, cs_name, trc_name,
                                  multiview=multiview, left_view_id=left_view_id)
            if not self.seek(0):  # revient au debut apres la frame de probe
                logger.warning("[HEVC] seek(0) post-probe refuse")
            logger.info(f"[HEVC] open OK: {os.path.basename(str(path))} "
                        f"{fr.width}x{fr.height} {name} stereo={hint} "
                        f"color={cs_name}/{trc_name} tb={self._tb_num}/{self._tb_den} "
                        f"multiview={multiview} left_view_id={left_view_id} "
                        f"primary_eye={_primary_eye}")
            return self.info
        except Exception as e:
            if _view_ids is not None:
                # Contrat spec §6 "Echec re-open avec view_ids -> retry silencieux
                # mono-vue (2D) + log" : jamais un refus (MV-1 a prouve le chemin
                # d'echec exact -- avcodec_send_packet err -22 des le 1er paquet
                # quand la vue demandee n'existe pas dans le VPS).
                logger.warning(f"[HEVC] multiview re-open en echec -> mono-vue: {e}")
                self.close()
                return self.open(path, allow_hw=allow_hw, _no_views=True)
            logger.info(f"[HEVC] open refuse: {e}")
            self.close()
            return None

    def _decode_next(self):
        """Avance jusqu'a la prochaine frame decodee dans self._frame. Retourne
        True ou None (EOF). Les erreurs mid-stream levent (comptees par l'appelant)."""
        AF, AC, AU = _lavf._AVFORMAT, _lavf._AVCODEC, _lavf._AVUTIL
        pkt = ctypes.cast(self._pkt, ctypes.POINTER(AVPacket))
        while True:
            r = AC.avcodec_receive_frame(self._dec, self._frame)
            if r == 0:
                fr = ctypes.cast(self._frame, ctypes.POINTER(AVFrame)).contents
                if not self._pts_verified and fr.pts != _AV_NOPTS:
                    # Garde ABI: le pts lu a l'offset pinne DOIT etre un pts envoye.
                    if fr.pts in self._sent_pts:
                        self._pts_verified = True
                    else:
                        logger.warning("[HEVC] offset pts AVFrame implausible -> refus")
                        raise RuntimeError("AVFrame.pts ABI check failed")
                return True
            if r == _AVERROR_EOF:
                return None
            if r != _AVERROR_EAGAIN:
                raise RuntimeError(f"receive_frame err {r}")
            # besoin d'un paquet de plus
            while True:
                if AF.av_read_frame(self._ctx, self._pkt) < 0:
                    AC.avcodec_send_packet(self._dec, None)   # drain EOF
                    break
                pk = pkt.contents
                if pk.stream_index != self._vidx or pk.size <= 0:
                    AC.av_packet_unref(self._pkt)
                    continue
                if pk.pts != _AV_NOPTS:
                    self._sent_pts.add(pk.pts)
                    if len(self._sent_pts) > 4096:
                        self._sent_pts.clear()   # borne memoire; la verif est deja faite
                s = AC.avcodec_send_packet(self._dec, self._pkt)
                AC.av_packet_unref(self._pkt)
                if s < 0 and s != _AVERROR_EAGAIN:
                    raise RuntimeError(f"send_packet err {s}")
                break

    def _on_get_format(self, ctx, fmts):
        """Callback avcodec (threads de decode) : retient d3d11 si propose, sinon
        le 1er format (SW). Minimal, thread-safe (logging Python l'est) ; AUCUN Qt."""
        i, first = 0, -1
        while fmts[i] != -1:
            if first < 0:
                first = fmts[i]
            if fmts[i] == _PIX["d3d11"]:
                self._hw_on = True
                return fmts[i]
            i += 1
        self._hw_on = False
        logger.info("[HEVC] d3d11 non propose par get_format -> SW")
        return first

    def hw_active(self):
        """True ssi get_format a REELLEMENT retenu la surface d3d11 (decode HW en
        cours). False = chemin SW (kill-switch, allow_hw=False, ou HW indispo)."""
        return bool(getattr(self, '_hw_on', False))

    def _probe_hw_bit_depth(self):
        """Transfert one-shot de la frame d3d11 courante vers une frame SW temp pour
        lire son format de download -> profondeur (nv12=8, p010le=10). La temp est
        liberee aussitot ; Task 10 reutilisera exactement ce chemin de transfert."""
        AU = _lavf._AVUTIL
        tmp = AU.av_frame_alloc()          # format defaut = AV_PIX_FMT_NONE(-1)
        if not tmp:
            return None
        try:
            if AU.av_hwframe_transfer_data(tmp, self._frame, 0) < 0:
                logger.info("[HEVC] av_hwframe_transfer_data (sonde) a echoue")
                return None
            f = ctypes.cast(tmp, ctypes.POINTER(AVFrame)).contents.format
            if f == _PIX["nv12"]:
                return 8
            if f == _PIX["p010le"]:
                return 10
            logger.info(f"[HEVC] format de transfert HW inattendu {f} -> refus")
            return None
        finally:
            p = ctypes.c_void_p(tmp)
            AU.av_frame_free(ctypes.byref(p))

    def seek(self, timestamp_ms):
        if not self._opened:
            return False
        AF, AC = _lavf._AVFORMAT, _lavf._AVCODEC
        ts = int(max(0.0, float(timestamp_ms)) * 1000.0)   # AV_TIME_BASE us, stream -1
        ok = AF.av_seek_frame(self._ctx, -1, ts, _AVSEEK_FLAG_BACKWARD) >= 0
        if ok:
            AC.avcodec_flush_buffers(self._dec)
            # Fix-5 (MV-5 final review): a view snapshot left over from BEFORE the seek
            # (self._mv_pending, spec §4 buffer) can never pair with the pts stream that
            # resumes after the flush -- keeping it guarantees a spurious "vue orpheline"
            # warning on the next read_view_pair() plus a stale-pair edge (pairing a
            # pre-seek view with a post-seek one on a pts coincidence). Drop it so
            # read_view_pair() starts clean from the landed position.
            self._mv_pending = None
        return ok

    def close(self):
        AF, AC, AU = (getattr(_lavf, n, None) for n in ("_AVFORMAT", "_AVCODEC", "_AVUTIL"))
        try:
            if self._frame and AU:
                p = ctypes.c_void_p(self._frame)
                AU.av_frame_free(ctypes.byref(p)); self._frame = None
            if self._swframe and AU:
                p = ctypes.c_void_p(self._swframe)
                AU.av_frame_free(ctypes.byref(p)); self._swframe = None
            if self._pkt and AC:
                p = ctypes.c_void_p(self._pkt)
                AC.av_packet_free(ctypes.byref(p)); self._pkt = None
            if self._dec and AC:
                p = ctypes.c_void_p(self._dec)
                AC.avcodec_free_context(ctypes.byref(p)); self._dec = None
            # Unref NOTRE reference device (avcodec_free_context a libere la sienne).
            hw = getattr(self, '_hw_ref', None)
            if hw is not None and AU and getattr(AU, 'av_buffer_unref', None):
                ref = hw if isinstance(hw, ctypes.c_void_p) else ctypes.c_void_p(hw)
                if ref.value:
                    AU.av_buffer_unref(ctypes.byref(ref))
            if self._ctx and AF:
                AF.avformat_close_input(ctypes.byref(self._ctx))
        except Exception:
            pass
        # Reset COMPLET de l'etat pour un re-open propre (revue Task 9, finding
        # #1) : le retry SW silencieux (open() -> close() -> open(allow_hw=False))
        # doit repartir d'un instance vierge, rien ne doit survivre de la session
        # HW avortee (surtout _hw_on/_hw_wanted, sinon hw_active() mentirait).
        self._opened = False
        self._sent_pts = set()
        self._pts_verified = False
        self._err_streak = 0
        self.failed = False
        self.info = None
        self._hw_wanted = False
        self._hw_on = False
        self._hw_ref = None
        self._keep_cb = None
        self._mv_pending = None
        self._mv_view_ids = None

    _SD_STEREO3D = None      # resolu au runtime via av_frame_side_data_name

    @classmethod
    def _resolve_stereo3d_sd_type(cls):
        """Trouve la valeur d'enum AV_FRAME_DATA_STEREO3D par son NOM (ABI-proof)."""
        if cls._SD_STEREO3D is not None:
            return cls._SD_STEREO3D
        AU = _lavf._AVUTIL
        for i in range(64):
            n = AU.av_frame_side_data_name(i)
            if n and n.lower() == b"stereo 3d":
                cls._SD_STEREO3D = i
                return i
        cls._SD_STEREO3D = -1
        return -1

    def _read_stereo_side_data(self, fr):
        """AVStereo3D {int type; int flags; ...}: SIDEBYSIDE=1->sbs, TOPBOTTOM=2->tab,
        FLAG_INVERT=1. AVFrameSideData: enum type (int, padde a 8) puis uint8_t *data @8.
        `fr` est ignore: la side-data est lue sur self._frame (frame courante)."""
        sd_type = self._resolve_stereo3d_sd_type()
        if sd_type < 0:
            return (None, False)
        AU = _lavf._AVUTIL
        sd = AU.av_frame_get_side_data(self._frame, sd_type)
        if not sd:
            return (None, False)
        data_ptr = ctypes.c_void_p.from_address(sd + 8).value
        if not data_ptr:
            return (None, False)
        s3d_type = ctypes.c_int.from_address(data_ptr).value
        s3d_flags = ctypes.c_int.from_address(data_ptr + 4).value
        mode = {1: 'sbs', 2: 'tab'}.get(s3d_type)
        return (mode, bool(s3d_flags & 1))

    @staticmethod
    def _opt_get_csv(dec, name):
        """av_opt_get(dec, name, AV_OPT_SEARCH_CHILDREN, &out) -> liste de tokens bytes
        non vides ("0,1" -> [b'0', b'1']), [] si absent/vide. Libere le buffer malloc
        via av_freep (contrat verifie MV-1)."""
        AU = _lavf._AVUTIL
        out = ctypes.c_void_p()
        ret = AU.av_opt_get(dec, name, _AV_OPT_SEARCH_CHILDREN, ctypes.byref(out))
        if ret != 0 or not out.value:
            return []
        val = ctypes.cast(out.value, ctypes.c_char_p).value
        AU.av_freep(ctypes.byref(out))
        return [p for p in val.split(b',') if p]

    @staticmethod
    def _map_left_view_id(ids, pos):
        """Mapping oeil -> view_id (spec §3). Source primaire: view_pos_available
        ("1,2") aligne positionnellement sur view_ids_available ("0,1") — pos '1' =
        l'oeil gauche, MV-1 verbatim (confirme par ffprobe primary_eye=left sur le
        sample). Repli (view_pos_available absent/longueur incoherente): la PREMIERE
        vue de view_ids_available = gauche (defaut spec §3), inversable ensuite par la
        plomberie inverted/swap_eyes existante."""
        if ids and pos and len(ids) == len(pos):
            for vid, p in zip(ids, pos):
                if p.strip() == b'1':
                    try:
                        return int(vid)
                    except ValueError:
                        break
        if ids:
            try:
                return int(ids[0])
            except ValueError:
                return 0
        return 0

    def _probe_multiview(self, path):
        """Sonde JETABLE la multiview-abilite du fichier (spec §3) : ouvre un
        AVFormatContext + AVCodecContext SEPARES (jamais self._ctx/self._dec), SANS
        poser l'option "threads" (defaut = single-thread).

        POURQUOI un contexte jetable et PAS self._dec (finding MV-2, verifie
        empiriquement, absent du rapport MV-1) : la session reelle pose TOUJOURS
        threads=0 (frame-threading auto, perf existante — non-regression). Sous
        frame-threading, "view_ids_available"/"view_pos_available" NE SE PEUPLENT
        JAMAIS (verifie : toujours b'' meme apres 10 frames decodees), qu'on lise
        l'option AVANT ou APRES avcodec_open2, AVANT ou APRES un decode, et MEME
        quand view_ids est explicitement pose. En single-thread (defaut, sans
        l'option "threads" posee), elles sont peuplees DES avcodec_open2 (MV-1
        verbatim). Le DECODAGE multiview lui-meme (avec view_ids pose) fonctionne
        parfaitement sous frame-threading (verifie : View-ID side-data + appariement
        pts corrects) — seule la LECTURE de ces deux options est masquee ; d'ou : une
        sonde single-thread dediee juste pour la detection + le mapping oeil, la
        session reelle gardant threads=0 pour la perf (test_seek_perf).

        Retourne (ids: list[bytes], left_view_id: int|None, primary_eye: str|None).
        ids vide (ou <2 valeurs) = pas multiview (left_view_id/primary_eye = None).
        Best-effort total : toute erreur -> ([], None, None), jamais de levee (la
        sonde ne doit jamais faire echouer un open() qui aurait pu reussir en mono-vue)."""
        AF, AC, AU = _lavf._AVFORMAT, _lavf._AVCODEC, _lavf._AVUTIL
        ctx = ctypes.c_void_p()
        dec = None
        pkt = None
        frame = None
        try:
            if AF.avformat_open_input(ctypes.byref(ctx), str(path).encode('utf-8'),
                                      None, None) < 0:
                return [], None, None
            try:
                if AF.avformat_find_stream_info(ctx, None) < 0:
                    return [], None, None
                vidx = AF.av_find_best_stream(ctx, _AVMEDIA_TYPE_VIDEO, -1, -1, None, 0)
                if vidx < 0:
                    return [], None, None
                fmt = ctypes.cast(ctx, ctypes.POINTER(AVFormatContext))
                sptr = fmt.contents.streams[vidx]
                cp_addr = ctypes.c_void_p.from_address(sptr + _lavf._OFF_CODECPAR).value
                if not cp_addr:
                    return [], None, None
                cp = AVCodecParameters.from_address(cp_addr)
                if (cp.codec_type != _AVMEDIA_TYPE_VIDEO
                        or AC.avcodec_get_name(cp.codec_id) != b"hevc"):
                    return [], None, None
                codec = AC.avcodec_find_decoder_by_name(b"hevc")
                if not codec:
                    return [], None, None
                dec = AC.avcodec_alloc_context3(codec)
                if not dec or AC.avcodec_parameters_to_context(dec, cp_addr) < 0:
                    return [], None, None
                # PAS de "threads" pose ici (delibere, cf. docstring) : defaut single-thread.
                if AC.avcodec_open2(dec, codec, None) < 0:
                    return [], None, None
                ids = self._opt_get_csv(dec, b"view_ids_available")
                pos = self._opt_get_csv(dec, b"view_pos_available")
                # Robustesse (MV-1: "la garder ne coute rien") + cross-check primary_eye :
                # decode 1 frame de probe (best-effort, jamais fatal a la sonde).
                pkt = AC.av_packet_alloc()
                frame = AU.av_frame_alloc()
                probe_ok = False
                if pkt and frame:
                    probe_ok = _decode_one_probe_frame(ctx, vidx, dec, pkt, frame)
                    if len(ids) < 2:
                        ids2 = self._opt_get_csv(dec, b"view_ids_available")
                        pos2 = self._opt_get_csv(dec, b"view_pos_available")
                        if len(ids2) >= 2:
                            ids, pos = ids2, pos2
                if len(ids) < 2:
                    return [], None, None
                left_view_id = self._map_left_view_id(ids, pos)
                eye = None
                if probe_ok and _HAS_STEREO3D_EYE_NAME:
                    try:
                        sd_type = self._resolve_stereo3d_sd_type()
                        if sd_type >= 0:
                            sd = AU.av_frame_get_side_data(frame, sd_type)
                            if sd:
                                data_ptr = ctypes.c_void_p.from_address(sd + 8).value
                                if data_ptr:
                                    eye_val = ctypes.c_int.from_address(data_ptr + 12).value
                                    eye_raw = AU.av_stereo3d_primary_eye_name(eye_val)
                                    eye = eye_raw.decode('ascii', 'replace') if eye_raw else None
                    except Exception:
                        pass   # cross-check informatif seulement (jamais decideur)
                logger.info(f"[HEVC] sonde multiview: ids={ids} pos={pos} "
                            f"left_view_id={left_view_id} primary_eye={eye}")
                return ids, left_view_id, eye
            finally:
                if frame:
                    p = ctypes.c_void_p(frame)
                    AU.av_frame_free(ctypes.byref(p))
                if pkt:
                    p = ctypes.c_void_p(pkt)
                    AC.av_packet_free(ctypes.byref(p))
                if dec:
                    p = ctypes.c_void_p(dec)
                    AC.avcodec_free_context(ctypes.byref(p))
                if ctx:
                    AF.avformat_close_input(ctypes.byref(ctx))
        except Exception as e:
            logger.info(f"[HEVC] sonde multiview echouee (best-effort, mono-vue suppose): {e}")
            return [], None, None

    def _read_color_metadata(self, cp_addr, width):
        """Resolve the color-space / transfer NAMES from AVCodecParameters (HDR10 plumbing).

        Reads the pinned offsets (major-62 only) and CROSS-CHECKS each value by name via
        av_color_space_name / av_color_transfer_name; a value whose name is implausible
        (offset didn't land where we think) or 'unspecified'/'reserved'/'unknown' falls back
        to the width heuristic (>=1280 wide -> '709', else '601'; trc -> None). Returns
        (color_space_name, color_trc_name), e.g. ('bt2020nc', 'smpte2084') or ('709', None)."""
        AC, AU = _lavf._AVCODEC, _lavf._AVUTIL
        cs_name = trc_name = None
        major_ok = (AC.avcodec_version() >> 16) == 62 and _OFF_COLOR_SPACE is not None
        if major_ok:
            try:
                cs_val = ctypes.c_int.from_address(cp_addr + _OFF_COLOR_SPACE).value
                trc_val = ctypes.c_int.from_address(cp_addr + _OFF_COLOR_TRC).value
                cs_raw = AU.av_color_space_name(cs_val)
                trc_raw = AU.av_color_transfer_name(trc_val)
                if cs_raw in _PLAUSIBLE_CS:
                    cs_name = cs_raw.decode('ascii', 'replace')
                else:
                    logger.warning(f"[HEVC] color_space name implausible "
                                   f"({cs_raw!r} val={cs_val}) -> heuristique largeur")
                if trc_raw in _PLAUSIBLE_TRC:
                    trc_name = trc_raw.decode('ascii', 'replace')
                else:
                    logger.warning(f"[HEVC] color_trc name implausible "
                                   f"({trc_raw!r} val={trc_val}) -> None")
            except Exception as e:
                logger.warning(f"[HEVC] lecture color metadata echouee -> heuristique: {e}")
        else:
            logger.info("[HEVC] avcodec major != 62 -> offsets color non appliques (heuristique)")
        # 'unspecified'/'reserved'/'unknown' (ou lecture ratee) -> heuristique.
        if cs_name in (None, 'unspecified', 'reserved', 'unknown'):
            cs_name = '709' if width >= 1280 else '601'
        if trc_name in (None, 'unspecified', 'reserved', 'unknown'):
            trc_name = None
        return cs_name, trc_name

    def _copy_plane(self, ptr, linesize, h, w, bps):
        """Copie INCONDITIONNELLE d'un plan de l'AVFrame vers numpy (bps=1|2).
        np.array(copy=True), PAS ascontiguousarray — ce dernier est un no-op
        quand linesize == w*bps (aucun padding, cas reel de tous les assets)
        et retournerait un ALIAS du pool de frames avcodec."""
        buf = np.ctypeslib.as_array(ptr, shape=(h, linesize))
        arr = np.array(buf[:, :w * bps], copy=True)
        return arr.view(np.uint16).reshape(h, w) if bps == 2 else arr

    def read_frame(self):
        """((Y,U,V), pts_ms) | None=EOF. >=5 erreurs consecutives -> failed=True."""
        if not self._opened or getattr(self, 'failed', False):
            return None
        while True:
            try:
                if self._decode_next() is None:
                    return None                      # EOF propre
            except Exception as e:
                self._err_streak += 1
                logger.warning(f"[HEVC] erreur decode ({self._err_streak}/5): {e}")
                if self._err_streak >= 5:
                    self.failed = True
                    return None
                continue
            self._err_streak = 0
            fr = ctypes.cast(self._frame, ctypes.POINTER(AVFrame)).contents
            if fr.format == _PIX["d3d11"]:
                # Frame HW (surface D3D11) : copy-back via transfert vers une frame SW
                # temp reutilisee (self._swframe), puis deinterleave NV12/P010 -> plans
                # Y/U/V contigus (meme contrat de sortie que le chemin SW).
                AU = _lavf._AVUTIL
                if self._swframe is None:
                    self._swframe = AU.av_frame_alloc()
                    if not self._swframe:
                        self._swframe = None
                        raise RuntimeError("alloc failed: av_frame_alloc (swframe)")
                AU.av_frame_unref(self._swframe)
                if AU.av_hwframe_transfer_data(self._swframe, self._frame, 0) < 0:
                    # Echec de transfert HW = classe device-lost/OOM, non transitoire:
                    # on remonte DIRECT (sans streak) au try du HevcDecodeThread ->
                    # decodeFailed -> le player retombe sur mpv. Choix assume.
                    raise RuntimeError("hwframe_transfer_data")
                sw = ctypes.cast(self._swframe, ctypes.POINTER(AVFrame)).contents
                # NV12/P010: data[0]=Y, data[1]=UV interleave. Deinterleave par vues
                # numpy stridees puis copie contigue (contrat de sortie inchange).
                # format HW gate une fois a l'open() (_probe_hw_bit_depth); ne change pas en cours de session.
                bps = 2 if sw.format == _PIX["p010le"] else 1
                w, h = fr.width, fr.height
                y = self._copy_plane(sw.data[0], sw.linesize[0], h, w, bps)
                uvbuf = np.ctypeslib.as_array(sw.data[1], shape=(h // 2, sw.linesize[1]))
                uv = np.array(uvbuf[:, :w * bps], copy=True)
                if bps == 2:
                    uv16 = uv.view(np.uint16).reshape(h // 2, w)
                    u = np.array(uv16[:, 0::2], copy=True)
                    v = np.array(uv16[:, 1::2], copy=True)
                else:
                    u = np.array(uv[:, 0::2], copy=True)
                    v = np.array(uv[:, 1::2], copy=True)
                pts = fr.pts
                pts_ms = (pts * 1000 * self._tb_num) // self._tb_den if pts != _AV_NOPTS else -1
                return ((y, u, v), pts_ms)
            bps = 2 if self.info.bit_depth == 10 else 1
            w, h = fr.width, fr.height
            y = self._copy_plane(fr.data[0], fr.linesize[0], h, w, bps)
            u = self._copy_plane(fr.data[1], fr.linesize[1], h // 2, w // 2, bps)
            v = self._copy_plane(fr.data[2], fr.linesize[2], h // 2, w // 2, bps)
            pts = fr.pts
            pts_ms = (pts * 1000 * self._tb_num) // self._tb_den if pts != _AV_NOPTS else -1
            return ((y, u, v), pts_ms)

    # ---------------------------------------------------------------------
    # MV-2: appariement des vues MV-HEVC (spec §4)
    # ---------------------------------------------------------------------

    _SD_VIEW_ID = None       # resolu au runtime via av_frame_side_data_name

    @classmethod
    def _resolve_view_id_sd_type(cls):
        """Trouve la valeur d'enum de la side-data frame 'View ID' par son NOM
        (ABI-proof, meme patron que _resolve_stereo3d_sd_type). Nom EXACT verifie
        MV-1 (verbatim): b'View ID' (payload int32 LE @ data_ptr)."""
        if cls._SD_VIEW_ID is not None:
            return cls._SD_VIEW_ID
        AU = _lavf._AVUTIL
        for i in range(64):
            n = AU.av_frame_side_data_name(i)
            if n and n.lower() == b"view id":
                cls._SD_VIEW_ID = i
                return i
        cls._SD_VIEW_ID = -1
        return -1

    def _read_view_id(self):
        """Lit la side-data 'View ID' (int32 LE @data_ptr, MV-1 verbatim) sur
        self._frame (frame courante). None si absente (mono-vue)."""
        sd_type = self._resolve_view_id_sd_type()
        if sd_type < 0:
            return None
        AU = _lavf._AVUTIL
        sd = AU.av_frame_get_side_data(self._frame, sd_type)
        if not sd:
            return None
        data_ptr = ctypes.c_void_p.from_address(sd + 8).value
        if not data_ptr:
            return None
        return ctypes.c_int32.from_address(data_ptr).value

    def _snapshot_current_view(self):
        """Copie les 3 plans (Y,U,V, contrat _copy_plane inconditionnel — pas d'alias
        du pool avcodec) + pts_ms + view_id de self._frame (frame fraichement decodee,
        AVANT tout decode suivant). Multiview = SW uniquement (spec §2: decodage
        logiciel — allow_hw force False au re-open), donc pas de branche d3d11 ici."""
        fr = ctypes.cast(self._frame, ctypes.POINTER(AVFrame)).contents
        bps = 2 if self.info.bit_depth == 10 else 1
        w, h = fr.width, fr.height
        y = self._copy_plane(fr.data[0], fr.linesize[0], h, w, bps)
        u = self._copy_plane(fr.data[1], fr.linesize[1], h // 2, w // 2, bps)
        v = self._copy_plane(fr.data[2], fr.linesize[2], h // 2, w // 2, bps)
        pts = fr.pts
        pts_ms = (pts * 1000 * self._tb_num) // self._tb_den if pts != _AV_NOPTS else -1
        return {'y': y, 'u': u, 'v': v, 'pts': pts, 'pts_ms': pts_ms,
                'view_id': self._read_view_id()}

    def _next_view_snapshot(self):
        """Avance jusqu'a la prochaine vue decodee et retourne son snapshot copie, ou
        None en EOF propre. Erreurs mid-stream (decode_next leve) : MEME contrat
        d'echec que read_frame (_err_streak, 5 consecutives -> failed=True), puis
        retente (ne casse pas la lecture pour une erreur transitoire)."""
        while True:
            try:
                if self._decode_next() is None:
                    return None
            except Exception as e:
                self._err_streak += 1
                logger.warning(f"[HEVC] erreur decode vue ({self._err_streak}/5): {e}")
                if self._err_streak >= 5:
                    self.failed = True
                    return None
                continue
            return self._snapshot_current_view()

    def _order_pair(self, a, b):
        """Assigne (left, right) via MediaInfo.left_view_id — JAMAIS l'ordre d'arrivee
        (spec §4 / lecon MVC [[mvc-view-pairing-saccade]]), meme si le sample observe
        alterne toujours (0, 1) dans cet ordre (MV-1 verbatim). Repli (view_id absent
        ou ne correspondant a aucun des deux, ex. side-data manquante) : ordre
        d'arrivee, documente."""
        left_id = self.info.left_view_id if self.info is not None else None
        if left_id is not None and b['view_id'] == left_id and a['view_id'] != left_id:
            a, b = b, a
        left = (a['y'], a['u'], a['v'])
        right = (b['y'], b['u'], b['v'])
        return (left, right, a['pts_ms'])

    def read_view_pair(self):
        """((Yl,Ul,Vl), (Yr,Ur,Vr), pts_ms) | None=EOF (spec §4).

        Buffer 1-slot : une vue decodee attend sa partenaire de PTS IDENTIQUE. Sur pts
        different, l'orpheline (le pending) est jetee — log `[HEVC] vue orpheline
        pts=...` + compteur _err_streak (MEME contrat que read_frame : 5 consecutives
        -> failed=True). Le nouvel arrivant devient le candidat en attente (il peut
        s'apparier au tour suivant). left/right assignes via _order_pair (left_view_id),
        pas l'ordre d'arrivee."""
        if not self._opened or getattr(self, 'failed', False):
            return None
        if self._mv_pending is None:
            self._mv_pending = self._next_view_snapshot()
            if self._mv_pending is None:
                return None                              # EOF propre, rien en attente
        while True:
            if getattr(self, 'failed', False):
                self._mv_pending = None
                return None
            nxt = self._next_view_snapshot()
            if nxt is None:
                # EOF alors qu'une vue attendait sa partenaire -> orpheline finale.
                logger.warning(f"[HEVC] vue orpheline pts={self._mv_pending['pts']} (EOF)")
                self._mv_pending = None
                self._err_streak += 1
                if self._err_streak >= 5:
                    self.failed = True
                return None
            if nxt['pts'] == self._mv_pending['pts']:
                pend = self._mv_pending
                self._mv_pending = None
                self._err_streak = 0
                return self._order_pair(pend, nxt)
            logger.warning(f"[HEVC] vue orpheline pts={self._mv_pending['pts']} "
                           f"(pts suivant={nxt['pts']})")
            self._err_streak += 1
            if self._err_streak >= 5:
                self.failed = True
                self._mv_pending = None
                return None
            self._mv_pending = nxt        # nxt devient le nouveau candidat en attente
