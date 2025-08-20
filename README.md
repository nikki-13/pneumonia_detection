# X-Ray Insight Lab

An AI-powered application for analyzing chest X-rays to detect pneumonia.

🔗 **Live Demo**: [https://nikki-13.github.io](https://nikki-13.github.io)

## Features

- Upload chest X-ray images for analysis
- Real-time AI prediction for pneumonia detection
- Confidence scores and detailed results
- Batch testing capabilities for model evaluation

## Tech Stack

- **Frontend**: React, TypeScript, Vite, Tailwind CSS, Shadcn UI
- **Backend**: FastAPI, Python
- **ML Models**: PyTorch, Ensemble of EfficientNet, SwinV2, and ResNet models

## Local Development Setup

### Prerequisites

- Node.js (v18 or later)
- Python 3.11 or later
- npm or yarn

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd x-ray-insight-lab
   ```

2. Install frontend dependencies:
   ```
   npm install
   ```

3. Set up the backend:
   ```
   cd server
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cd ..
   ```

### Running the Application

You can use the provided scripts to run both frontend and backend:

#### First-time setup:

```
chmod +x start_app.sh
./start_app.sh
```

This will install all dependencies and start the application. The first run may take some time as it installs all required packages.

#### Quick start (after initial setup):

For subsequent runs, you can use the quick start script which skips dependency checks and installations:

```
chmod +x quick_start.sh
./quick_start.sh
```

#### Running components separately:

1. Start the backend server:
   ```
   cd server
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   python run.py
   ```

2. In a new terminal, start the frontend:
   ```
   npm run dev
   ```

The application will be available at:
- Frontend: http://localhost:5173 or http://localhost:8080
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Model Information

The application uses an ensemble of three deep learning models:
- EfficientNet B0
- SwinV2 Tiny
- ResNet50

These models are combined to provide more accurate predictions than any single model.

## API Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Upload and analyze an X-ray image
- `GET /test`: Run batch tests on the model using test images

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are installed correctly
2. Check that the model files are in the correct location (src/model/)
3. Verify that the backend server is running and accessible
4. Check the console for any error messages

## Deployment

### GitHub Pages

The frontend is automatically deployed to GitHub Pages on every push to the main branch.

**Live URL**: [https://nikki-13.github.io](https://nikki-13.github.io)

#### Manual Deployment

To deploy manually:

```bash
# Using the deployment script
./deploy.sh

# Or using npm commands
npm run build
npm run deploy
```

#### GitHub Actions

The project includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) that automatically:
- Builds the project on every push to main
- Deploys to GitHub Pages
- Serves the application at the live URL

### Local Deployment

For local deployment testing:

```bash
npm run build
npm run preview
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.
