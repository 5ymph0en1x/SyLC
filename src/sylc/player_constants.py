"""Small cross-module constants for the SyLC player shell."""

# Output keys emitted by the stereo presentation selector. ``auto`` denotes
# the absence of an explicit pick and is therefore intentionally excluded.
PRESENTATION_KEYS = ('mvc', 'sbs', 'tab', 'dual', 'glasses')


__all__ = ['PRESENTATION_KEYS']
