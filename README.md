# CardioVortex

Cardiac electrophysiology spiral-rotor and LEAP defibrillation simulation engine.

Simulates 2D myocardial reaction-diffusion excitable media (Aliev–Panfilov / FitzHugh–Nagumo),
tracks phase singularities (rotor cores) in ventricular fibrillation, and evaluates
low-energy anti-fibrillation pacing (LEAP) protocols.

## Install

```bash
uv sync --extra dev
```

## Tests

```bash
uv run pytest -q
```

## License

MIT
