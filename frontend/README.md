# Capsule Movie AI - Vue.js Frontend

A modern, professional Vue.js frontend inspired by SkyReels.ai design for the Capsule Movie AI video generation platform.

## 🌟 Features

- **Modern Vue.js 3** with Composition API
- **Professional Dark Theme** inspired by SkyReels.ai
- **Responsive Design** with mobile-first approach
- **Real-time Progress Tracking** for video generation
- **Interactive Studio** with multiple generation modes
- **Glass Morphism UI** with beautiful animations
- **Tailwind CSS** for rapid styling

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- npm or yarn

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# The app will be available at http://localhost:3000
```

### Build for Production

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## 🎨 Design Philosophy

This frontend is **inspired by** the elegant design of SkyReels.ai while being completely original code. Key design elements:

- **Dark Theme**: Professional dark background with gradient accents
- **Glass Effects**: Frosted glass components with backdrop blur
- **Gradient Branding**: Blue to purple gradients matching AI aesthetics
- **Interactive Elements**: Smooth transitions and hover effects
- **Typography**: Clean Inter font for readability

## 📱 Pages & Components

### Pages
- **Home** (`/`) - Landing page with hero section and feature preview
- **Studio** (`/studio`) - Main video generation interface
- **Features** (`/features`) - Detailed feature showcase
- **Pricing** (`/pricing`) - Subscription plans and FAQ

### Key Components
- **Navigation** - Fixed glass navigation with mobile menu
- **AuthModal** - Social login and signup modal
- **LoadingOverlay** - Global loading state with progress
- **Footer** - Links and company information

## 🛠️ Technical Stack

- **Vue.js 3** - Progressive JavaScript framework
- **Vue Router 4** - Client-side routing
- **Pinia** - State management
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Axios** - HTTP client for API calls

## 🎯 Studio Features

### Text-to-Video Generation
- Multi-model selection (540P/720P)
- Duration control (4s to 30s+)
- Advanced parameters (guidance scale, quality steps)
- Prompt enhancement option
- Real-time progress tracking

### Image-to-Video Animation
- Drag & drop image upload
- Animation description input
- Quality and duration controls
- Preview generation

### Long-Form Video (Diffusion Forcing)
- Infinite-length video creation
- Asynchronous generation options
- Frame control settings
- Advanced AI parameters

## 🎨 Customization

### Colors
The color scheme can be customized in `tailwind.config.js`:

```js
colors: {
  'capsule-dark': '#0f0f23',
  'capsule-blue': '#3b82f6',
  'capsule-purple': '#8b5cf6',
  'capsule-gold': '#fbbf24',
}
```

### Animations
Custom animations are defined in `src/style.css`:
- Float animation for background elements
- Pulse glow for interactive elements
- Smooth page transitions

## 🔌 API Integration

The frontend is designed to integrate with the SkyReels V2 Python backend:

```js
// Example API call structure
import axios from 'axios'

const generateVideo = async (params) => {
  const response = await axios.post('/api/generate-video', params)
  return response.data
}
```

## 📦 Project Structure

```
frontend/
├── src/
│   ├── components/          # Reusable Vue components
│   │   ├── Navigation.vue   # Main navigation
│   │   ├── AuthModal.vue    # Authentication modal
│   │   ├── Footer.vue       # Site footer
│   │   └── LoadingOverlay.vue # Loading states
│   ├── views/              # Page components
│   │   ├── Home.vue        # Landing page
│   │   ├── Studio.vue      # Video generation studio
│   │   ├── Features.vue    # Feature showcase
│   │   └── Pricing.vue     # Pricing and plans
│   ├── router/             # Vue Router configuration
│   ├── App.vue            # Root component
│   ├── main.js            # Application entry point
│   └── style.css          # Global styles
├── package.json           # Dependencies and scripts
├── vite.config.js        # Vite configuration
├── tailwind.config.js    # Tailwind CSS config
└── postcss.config.js     # PostCSS config
```

## 🚀 Deployment

### Vercel (Recommended)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

### Netlify
```bash
# Build
npm run build

# Deploy dist/ folder to Netlify
```

### Docker
```dockerfile
FROM node:18-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
```

## 🔄 Development Workflow

1. **Start Development Server**
   ```bash
   npm run dev
   ```

2. **Make Changes**
   - Edit components in `src/`
   - Hot reload automatically updates

3. **Test Build**
   ```bash
   npm run build
   npm run preview
   ```

4. **Deploy**
   ```bash
   git add .
   git commit -m "Update frontend"
   git push origin main
   ```

## 📄 License

This project is licensed under the MIT License - see the main repository license for details.

## 🙏 Acknowledgments

- **Inspired by** SkyReels.ai's elegant design
- **Powered by** SkyReels V2 technology
- **Built with** Vue.js ecosystem

---

**🎬 Capsule Movie AI** - Professional video generation at your fingertips!