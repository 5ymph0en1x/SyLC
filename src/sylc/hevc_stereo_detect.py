# -*- coding: utf-8 -*-
"""Detection du packing stereo d'un flux HEVC. Priorite: side-data (SEI
frame-packing / Matroska StereoMode, exposes par avcodec en AVStereo3D)
> tokens du nom de fichier > None (l'UI tranche). Full vs half = ratio par oeil."""
import os
import re

# DEVIATION (documentee dans task-5-report.md): Python `\b` traite `_` comme
# caractere de mot (\w inclut _), donc `\bfsbs\b` NE matche PAS "fsbs_1080p"
# (aucune frontiere entre "s" et "_") et `\bhou\b` NE matche PAS "film_hou_1080p"
# (encadre par des "_" des deux cotes). Ce bug cassait les propres tests du
# brief (Step 1, cas 3 et 4). Fix: frontiere custom via lookaround qui EXCLUT
# underscore du jeu de caracteres "mot", donc traite _ comme un separateur au
# meme titre que . / espace / -.
_B0, _B1 = r'(?<![A-Za-z0-9])', r'(?![A-Za-z0-9])'

# (token regex, mode, half) — ordre = priorite (les tokens longs d'abord)
_TOKENS = [
    (_B0 + r'fsbs' + _B1 + r'|full[ ._-]?sbs', 'sbs', False),
    (_B0 + r'ftab' + _B1 + r'|' + _B0 + r'fou' + _B1 + r'|full[ ._-]?(tab|ou)', 'tab', False),
    (_B0 + r'hsbs' + _B1 + r'|half[ ._-]?sbs', 'sbs', True),
    (_B0 + r'htab' + _B1 + r'|' + _B0 + r'hou' + _B1 + r'|half[ ._-]?(tab|ou)', 'tab', True),
    (_B0 + r'sbs' + _B1, 'sbs', None),          # half deduit du ratio
    (_B0 + r'tab' + _B1 + r'|' + _B0 + r'ou' + _B1, 'tab', None),
]


def _from_filename(path):
    name = os.path.basename(str(path)).lower()
    for rx, mode, half in _TOKENS:
        if re.search(rx, name):
            return mode, bool(half)
    return None, False


def _half_from_ratio(mode, width, height):
    """Full-SBS 16:9 -> frame 32:9 (~3.55); half-SBS -> ~16:9 (1.78). TAB inverse."""
    r = width / max(1, height)
    if mode == 'sbs':
        return r < 2.5          # 3.55=full, 1.78=half
    return r > 1.2              # tab: 0.89=full (16:18), 1.78=half


def detect(path, media_info):
    """-> (mode|'sbs'|'tab'|None, half, inverted). media_info = MediaInfo de la source."""
    if media_info is not None and media_info.stereo_hint in ('sbs', 'tab'):
        mode = media_info.stereo_hint
        return (mode, _half_from_ratio(mode, media_info.width, media_info.height),
                bool(media_info.stereo_inverted))
    mode, half = _from_filename(path)
    if mode is None:
        return (None, False, False)
    if half is False and re.search(_B0 + r'sbs' + _B1 + r'|' + _B0 + r'tab' + _B1
                                    + r'|' + _B0 + r'ou' + _B1,
                                   os.path.basename(str(path)).lower()) \
            and media_info is not None:
        half = _half_from_ratio(mode, media_info.width, media_info.height)
    return (mode, half, False)
