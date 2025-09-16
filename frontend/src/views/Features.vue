<template>
  <div class="min-h-screen pt-16">
    <!-- Header -->
    <div class="section-padding bg-gradient-to-r from-capsule-dark to-gray-900">
      <div class="container-max text-center">
        <h1 class="text-4xl font-bold text-white mb-4">Powerful Features</h1>
        <p class="text-xl text-white/70 max-w-3xl mx-auto">
          Discover the advanced capabilities that make Capsule Movie AI the premier platform for AI video generation
        </p>
      </div>
    </div>

    <!-- Main Features -->
    <div class="section-padding">
      <div class="container-max">
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center mb-20">
          <!-- Text-to-Video Feature -->
          <div>
            <div class="text-5xl mb-6">📝</div>
            <h2 class="text-3xl font-bold text-white mb-4">Advanced Text-to-Video</h2>
            <p class="text-white/70 text-lg mb-6">
              Transform any text description into stunning, professional-quality videos. Our AI understands context, 
              emotions, and visual storytelling to create exactly what you envision.
            </p>
            <ul class="space-y-3">
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                Multiple model sizes (1.3B to 14B parameters)
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-purple rounded-full mr-3"></span>
                Up to 720P resolution output
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-gold rounded-full mr-3"></span>
                Intelligent prompt enhancement
              </li>
            </ul>
          </div>
          <div class="card">
            <div class="aspect-video bg-gradient-to-br from-capsule-blue/20 to-capsule-purple/20 rounded-lg flex items-center justify-center">
              <div class="text-center text-white/60">
                <div class="w-20 h-20 mx-auto mb-4 bg-white/10 rounded-full flex items-center justify-center">
                  📝
                </div>
                <p>Text-to-Video Demo</p>
              </div>
            </div>
          </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center mb-20">
          <!-- Image-to-Video Feature -->
          <div class="order-2 lg:order-1 card">
            <div class="aspect-video bg-gradient-to-br from-capsule-purple/20 to-capsule-gold/20 rounded-lg flex items-center justify-center">
              <div class="text-center text-white/60">
                <div class="w-20 h-20 mx-auto mb-4 bg-white/10 rounded-full flex items-center justify-center">
                  🖼️
                </div>
                <p>Image-to-Video Demo</p>
              </div>
            </div>
          </div>
          <div class="order-1 lg:order-2">
            <div class="text-5xl mb-6">🖼️</div>
            <h2 class="text-3xl font-bold text-white mb-4">Image Animation</h2>
            <p class="text-white/70 text-lg mb-6">
              Bring static images to life with realistic motion and seamless transitions. Perfect for 
              creating dynamic content from existing artwork, photos, or designs.
            </p>
            <ul class="space-y-3">
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                Automatic image preprocessing and optimization
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-purple rounded-full mr-3"></span>
                Preserves original image quality and style
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-gold rounded-full mr-3"></span>
                Natural motion generation
              </li>
            </ul>
          </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center mb-20">
          <!-- Diffusion Forcing Feature -->
          <div>
            <div class="text-5xl mb-6">🎞️</div>
            <h2 class="text-3xl font-bold text-white mb-4">Infinite-Length Videos</h2>
            <p class="text-white/70 text-lg mb-6">
              Revolutionary Diffusion Forcing technology enables the creation of videos of any length. 
              Generate everything from short clips to full-length films.
            </p>
            <ul class="space-y-3">
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                Videos up to 60+ seconds and beyond
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-purple rounded-full mr-3"></span>
                Seamless frame transitions
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-gold rounded-full mr-3"></span>
                Temporal consistency across the entire video
              </li>
            </ul>
          </div>
          <div class="card">
            <div class="aspect-video bg-gradient-to-br from-capsule-gold/20 to-capsule-blue/20 rounded-lg flex items-center justify-center">
              <div class="text-center text-white/60">
                <div class="w-20 h-20 mx-auto mb-4 bg-white/10 rounded-full flex items-center justify-center">
                  🎞️
                </div>
                <p>Long-Form Video Demo</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Technical Features Grid -->
    <div class="section-padding bg-gray-900/50">
      <div class="container-max">
        <div class="text-center mb-16">
          <h2 class="text-4xl font-bold text-white mb-4">Technical Capabilities</h2>
          <p class="text-xl text-white/70">Cutting-edge technology powering next-generation video creation</p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          <div v-for="feature in technicalFeatures" :key="feature.name" class="card text-center">
            <div class="text-3xl mb-4">{{ feature.icon }}</div>
            <h3 class="text-lg font-semibold text-white mb-2">{{ feature.name }}</h3>
            <p class="text-white/70 text-sm">{{ feature.description }}</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Performance Stats -->
    <div class="section-padding">
      <div class="container-max">
        <div class="text-center mb-16">
          <h2 class="text-4xl font-bold text-white mb-4">Industry-Leading Performance</h2>
          <p class="text-xl text-white/70">Benchmarked against the world's best video generation models</p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div class="text-center">
            <div class="text-4xl font-bold gradient-text mb-2">83.9%</div>
            <div class="text-white/80 font-medium">VBench Total Score</div>
            <div class="text-white/60 text-sm">Industry Leading</div>
          </div>
          <div class="text-center">
            <div class="text-4xl font-bold gradient-text mb-2">3.14</div>
            <div class="text-white/80 font-medium">Human Evaluation</div>
            <div class="text-white/60 text-sm">Best in Class</div>
          </div>
          <div class="text-center">
            <div class="text-4xl font-bold gradient-text mb-2">14B</div>
            <div class="text-white/80 font-medium">Parameters</div>
            <div class="text-white/60 text-sm">Maximum Model</div>
          </div>
          <div class="text-center">
            <div class="text-4xl font-bold gradient-text mb-2">720P</div>
            <div class="text-white/80 font-medium">Max Resolution</div>
            <div class="text-white/60 text-sm">HD Quality</div>
          </div>
        </div>
      </div>
    </div>

    <!-- CTA Section -->
    <div class="section-padding bg-gradient-to-r from-capsule-blue/10 to-capsule-purple/10">
      <div class="container-max text-center">
        <h2 class="text-4xl font-bold text-white mb-4">Experience the Future of Video Creation</h2>
        <p class="text-xl text-white/70 mb-8 max-w-2xl mx-auto">
          Join the next generation of content creators using AI to produce stunning videos
        </p>
        <router-link to="/studio" class="btn-primary text-lg px-8 py-4">
          Try All Features
        </router-link>
      </div>
    </div>
  </div>
</template>

<script setup>
const technicalFeatures = [
  {
    name: 'Diffusion Forcing',
    icon: '⚡',
    description: 'Revolutionary architecture enabling infinite-length video generation with temporal consistency'
  },
  {
    name: 'Multi-GPU Support',
    icon: '🚀',
    description: 'Distributed inference with xDiT USP acceleration for faster generation times'
  },
  {
    name: 'TeaCache Optimization',
    icon: '💾',
    description: 'Intelligent caching system that reduces inference time while maintaining quality'
  },
  {
    name: 'Prompt Enhancement',
    icon: '✨',
    description: 'AI-powered prompt optimization using Qwen2.5-32B for better results'
  },
  {
    name: 'Frame Control',
    icon: '🎯',
    description: 'Precise control over start and end frames for seamless video transitions'
  },
  {
    name: 'Video Extension',
    icon: '🔄',
    description: 'Extend existing videos with new content while maintaining consistency'
  },
  {
    name: 'Batch Processing',
    icon: '📦',
    description: 'Generate multiple videos simultaneously with queue management'
  },
  {
    name: 'Quality Control',
    icon: '🎨',
    description: 'Advanced quality parameters for fine-tuning output to your needs'
  },
  {
    name: 'Model Variants',
    icon: '🧠',
    description: 'Choose from multiple model sizes optimized for speed or quality'
  }
]
</script>