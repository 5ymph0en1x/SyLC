# -*- coding: utf-8 -*-
r"""Instrumente la chaine PGS de bout en bout sur un fichier reel.

La chaine complete est :

  PMT (stream_type 0x90)  ->  get_subtitle_pids()
      -> set_subtitle_pid(PID)          [selectedSubtitlePid_ cote C++]
      -> collectSubtitlePacket()        [reassemblage PES -> subtitleQueue_]
      -> has_subtitle_data()/read_subtitle_block()
      -> pgsDataReady                   [Qt, cote mvc_decoder]
      -> SubtitleManager.on_pgs_data    [parseur PGS]
      -> subtitle_changed               [-> widget.set_subtitle]

Chaque etage peut echouer EN SILENCE : la boucle de decodage avale les erreurs
du polling (`except Exception: pass`), et une file vide est indistinguable d'un
PID qui ne correspond a rien. Cette sonde rend chaque etage visible.

Usage :
    .venv\Scripts\python.exe tools\probe_pgs_chain.py <base.m2ts> [dependent.ssif]
    .venv\Scripts\python.exe tools\probe_pgs_chain.py <fichier.mkv>

Optionnel : --pid 0x1200 pour forcer un PID, --frames 900 pour lire plus loin.
"""
import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'runtime'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base')
    ap.add_argument('dependent', nargs='?')
    ap.add_argument('--pid', default=None, help="PID a selectionner (ex 0x1200)")
    ap.add_argument('--frames', type=int, default=600,
                    help="nombre de paires d'images a lire avant d'abandonner")
    args = ap.parse_args()

    import mvc_demuxer_cpp as M
    print(f'== ETAGE 0 : ouverture ==')
    if args.dependent:
        dmx = M.MVCSSIFDemuxer()
        ok = dmx.open_dual(args.base, args.dependent)
        print(f'   open_dual -> {ok}')
    else:
        ext = os.path.splitext(args.base)[1].lower()
        if ext in ('.mkv', '.mk3d'):
            dmx = M.MVCMatroskaDemuxer() if hasattr(M, 'MVCMatroskaDemuxer') else M.MKVDemuxer()
        else:
            dmx = M.MVCSSIFDemuxer()
        ok = dmx.open(args.base)
        print(f'   open -> {ok}')
    if not ok:
        sys.exit('   ECHEC ouverture : la chaine s arrete ici.')

    print('\n== ETAGE 1 : pistes / PID annonces ==')
    pids = []
    if hasattr(dmx, 'get_subtitle_pids'):
        pids = list(dmx.get_subtitle_pids())
        print(f'   get_subtitle_pids() -> {[hex(p) for p in pids]}')
        if not pids:
            print('   !! Le PMT n annonce AUCUN flux stream_type 0x90.')
            print('      -> Le lecteur ne pourra jamais selectionner de PGS ici.')
    if hasattr(dmx, 'get_subtitle_tracks'):
        tracks = dmx.get_subtitle_tracks()
        print(f'   get_subtitle_tracks() -> {len(tracks)} piste(s)')
        for t in tracks[:12]:
            print(f'      trackNumber={t.get("trackNumber")} codec={t.get("codecId")} '
                  f'PGS={t.get("isPGS")} lang={t.get("language")!r}')
        pids = pids or [t.get('trackNumber') for t in tracks if t.get('isPGS')]

    if not hasattr(dmx, 'has_subtitle_data'):
        sys.exit('   !! Ce demuxeur n expose PAS has_subtitle_data() : '
                 'le polling Python ne fera JAMAIS rien (echec silencieux).')

    chosen = int(args.pid, 0) if args.pid else (pids[0] if pids else 0)
    if not chosen:
        sys.exit('   Aucun PID/piste a selectionner : arret.')
    print(f'\n== ETAGE 2 : selection du PID/piste {chosen} (0x{chosen:04X}) ==')
    if hasattr(dmx, 'set_subtitle_track'):
        dmx.set_subtitle_track(chosen)
    else:
        dmx.set_subtitle_pid(chosen)

    print(f'\n== ETAGE 3 : lecture de {args.frames} paires, collecte des blocs ==')
    blocks, first_pts, total_bytes = 0, None, 0
    frames = 0
    samples = []
    for i in range(args.frames):
        res = dmx.read_next_frame_pair()
        ok_read = res[0] if isinstance(res, tuple) else res
        if not ok_read:
            print(f'   fin de lecture apres {i} paires')
            break
        frames += 1
        while dmx.has_subtitle_data():
            got, blk = dmx.read_subtitle_block()
            if not got or blk is None:
                break
            blocks += 1
            data = bytes(blk['data'])
            total_bytes += len(data)
            if first_pts is None:
                first_pts = blk['timestampMs'] / 1000.0
            if len(samples) < 3:
                samples.append((blk['timestampMs'], data[:16].hex()))
    print(f'   images lues        : {frames}')
    print(f'   blocs PGS recoltes : {blocks} ({total_bytes} octets)')
    if blocks:
        print(f'   premier PTS        : {first_pts:.3f}s')
        for ts, head in samples:
            print(f'      bloc ts={ts}ms  debut={head}')
    else:
        print('   !! AUCUN bloc : soit le PID ne correspond a aucun paquet du flux,')
        print('      soit la collecte n est pas atteinte sur ce chemin de lecture.')
        print('      -> relancer avec --pid <autre valeur> pour discriminer.')
        return

    print('\n== ETAGE 4 : parseur PGS ==')
    try:
        from sylc.pgs_subtitle_parser import PGSParser
    except Exception as e:
        sys.exit(f'   import PGSParser impossible : {e}')
    p = PGSParser()
    fed = rendered = 0
    if hasattr(p, 'feed_pes_packet'):
        # rejouer les blocs deja consommes n est pas possible : on relit depuis le debut
        print('   (le parseur consomme un flux ; relance de la lecture pour l alimenter)')
    print('   API disponible :', [a for a in dir(p) if 'feed' in a or 'parse' in a][:6])
    print('\nResume : la chaine est intacte jusqu a l etage 3 inclus.')


if __name__ == '__main__':
    main()
