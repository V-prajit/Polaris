# ParkSight Starter Kit & Dashboard

Estimate parking capacity from satellite imagery using computer vision and machine learning. Built for **Hacklytics 2026**.
Includes a Next.js frontend for dashboard visualization.

## Quick Start (Backend)

```bash
# Clone and install
git clone <repo-url> && cd parksight-starter-kit
pip install -r requirements.txt

# Run the CLI baseline
python examples/run_baseline.py --address "Georgia Tech, Atlanta, GA"
```

## Quick Start (Frontend)

First, run the development server:
```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Project Structure

```
parksight-starter-kit/
├── README.md                        # You are here
├── LICENSE                          # MIT
├── requirements.txt                 # pip install -r requirements.txt
├── config.json                      # Tunable CV parameters
├── parksight/                       # Python package
├── src/                             # Next.js frontend code
├── api/                             # Local proxy/backend logic
└── package.json                     # Frontend dependencies
```

## License
MIT
