# Interface design

The application uses native Tk and ttk widgets with a shared visual system in
`ui_theme.py`. This keeps the existing event loop, camera lifecycle, measurement
callbacks, and deployment requirements unchanged.

## Design direction

- Dark neutral surfaces keep the camera images visually dominant.
- Teal identifies the active/healthy state; blue identifies inspection actions;
  red is reserved for reset, stop, and error actions.
- Information is grouped into bordered cards with an 8–16 pixel spacing rhythm.
- Button labels describe operator actions directly and use large click targets.
- Status always uses both text and color, so color is never the only signal.
- The same tokens and control styles are used on Dashboard, System Settings,
  Inspection Profile, Color Mask Setup, and Camera Setup.

## Performance

The theme introduces no timers, transparency effects, image assets, or continuous
animation. Hover feedback runs only on pointer enter/leave events. Camera preview
frequency and all processing behavior remain controlled by `config.json`.

## Reference guidance

- Microsoft Windows design principles:
  https://learn.microsoft.com/en-us/windows/apps/design/design-principles
- Microsoft content layout and spacing:
  https://learn.microsoft.com/en-us/windows/apps/design/basics/content-basics
- Microsoft inclusive software guidance:
  https://learn.microsoft.com/en-us/windows/apps/design/accessibility/designing-inclusive-software
