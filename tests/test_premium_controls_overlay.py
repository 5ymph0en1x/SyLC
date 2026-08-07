"""UI regressions for the compact premium controls menus."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from sylc.premium_controls_overlay import PremiumControlsOverlay


def test_synth3d_diagnostics_wrap_without_widening_menu():
    app = QApplication.instance() or QApplication([])
    overlay = PremiumControlsOverlay()
    status = (
        "TensorRT · Quality 756x322px · 23.7 depth fps · 68832 ms · "
        "2 surfaces · Lab 0.0% / p95 actif 0% / bord protégé 0.6% / "
        "champ source 192x81 · matte 12.2/24.0 fps · 59 ms @640p · "
        "contour 51%/16.4 ms / sparse 3.8% / rejet local 21.9%"
    )

    overlay.set_synth3d_status(status)
    menu = overlay.synth3d_menu
    menu.ensurePolished()
    menu.adjustSize()

    assert menu.minimumWidth() == overlay._SYNTH3D_MENU_WIDTH
    assert menu.maximumWidth() == overlay._SYNTH3D_MENU_WIDTH
    assert menu.width() == overlay._SYNTH3D_MENU_WIDTH
    assert overlay.synth3d_status_action.text() == ""
    assert overlay.synth3d_status_label.wordWrap()
    assert overlay.synth3d_status_label.text() == status
    assert overlay.synth3d_status_label.toolTip() == status
    assert overlay.synth3d_status_action.toolTip() == status

    overlay.close()
    overlay.deleteLater()
    app.processEvents()
