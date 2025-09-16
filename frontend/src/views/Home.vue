<template>
  <div class="min-h-screen">
    <!-- Hero Section -->
    <section class="relative overflow-hidden">
      <!-- Background gradient -->
      <div class="absolute inset-0 bg-gradient-to-br from-capsule-dark via-gray-900 to-capsule-dark"></div>
      
      <!-- Animated background elements -->
      <div class="absolute inset-0 opacity-20">
        <div class="absolute top-1/4 left-1/4 w-64 h-64 bg-capsule-blue rounded-full mix-blend-multiply filter blur-xl animate-float"></div>
        <div class="absolute top-3/4 right-1/4 w-64 h-64 bg-capsule-purple rounded-full mix-blend-multiply filter blur-xl animate-float" style="animation-delay: 2s;"></div>
        <div class="absolute bottom-1/4 left-1/3 w-64 h-64 bg-capsule-gold rounded-full mix-blend-multiply filter blur-xl animate-float" style="animation-delay: 4s;"></div>
      </div>

      <div class="relative container-max section-padding">
        <div class="text-center pt-16">
          <!-- Main Headline -->
          <h1 class="text-5xl md:text-7xl font-bold text-white leading-tight mb-6">
            Create 
            <span class="gradient-text">Infinite</span>
            <br />
            Length Films
          </h1>
          
          <!-- Subtitle -->
          <p class="text-xl md:text-2xl text-white/80 mb-8 max-w-3xl mx-auto leading-relaxed">
            Advanced AI video generation powered by Diffusion Forcing technology. 
            Transform your ideas into professional-quality videos in minutes.
          </p>

          <!-- CTA Buttons -->
          <div class="flex flex-col sm:flex-row items-center justify-center gap-4 mb-12">
            <router-link to="/studio" class="btn-primary text-lg px-8 py-4">
              🎬 Start Creating
            </router-link>
            <button @click="scrollToDemo" class="btn-ghost text-lg px-8 py-4">
              ▶️ Watch Demo
            </button>
          </div>

          <!-- Stats -->
          <div class="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-2xl mx-auto">
            <div class="text-center">
              <div class="text-3xl font-bold gradient-text">60s+</div>
              <div class="text-white/60">Video Length</div>
            </div>
            <div class="text-center">
              <div class="text-3xl font-bold gradient-text">720P</div>
              <div class="text-white/60">Max Resolution</div>
            </div>
            <div class="text-center">
              <div class="text-3xl font-bold gradient-text">14B</div>
              <div class="text-white/60">Parameters</div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Demo Section -->
    <section ref="demoSection" class="section-padding bg-gray-900/50">
      <div class="container-max">
        <div class="text-center mb-12">
          <h2 class="text-4xl font-bold text-white mb-4">See It In Action</h2>
          <p class="text-xl text-white/70">Experience the power of AI video generation</p>
        </div>

        <!-- Video Demo Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          <div class="card">
            <div class="aspect-video bg-gradient-to-br from-capsule-blue/20 to-capsule-purple/20 rounded-lg mb-4 flex items-center justify-center">
              <div class="text-center text-white/60">
                <div class="w-16 h-16 mx-auto mb-2 bg-white/10 rounded-full flex items-center justify-center">
                  ▶️
                </div>
                <p>Text-to-Video Demo</p>
              </div>
            </div>
            <h3 class="text-lg font-semibold text-white mb-2">From Text to Cinema</h3>
            <p class="text-white/70">Generate professional videos from simple text descriptions</p>
          </div>

          <div class="card">
            <div class="aspect-video bg-gradient-to-br from-capsule-purple/20 to-capsule-gold/20 rounded-lg mb-4 flex items-center justify-center">
              <div class="text-center text-white/60">
                <div class="w-16 h-16 mx-auto mb-2 bg-white/10 rounded-full flex items-center justify-center">
                  🖼️
                </div>
                <p>Image-to-Video Demo</p>
              </div>
            </div>
            <h3 class="text-lg font-semibold text-white mb-2">Bring Images to Life</h3>
            <p class="text-white/70">Animate static images with realistic motion and transitions</p>
          </div>
        </div>

        <!-- Quick Try Section -->
        <div class="card max-w-4xl mx-auto">
          <div class="text-center mb-6">
            <h3 class="text-2xl font-bold text-white mb-2">Try It Now</h3>
            <p class="text-white/70">Generate a quick sample video</p>
          </div>

          <div class="space-y-4">
            <div>
              <label class="block text-sm font-medium text-white/80 mb-2">Video Description</label>
              <textarea 
                v-model="demoPrompt"
                class="input-field h-24 resize-none"
                placeholder="Describe the video you want to create..."
              ></textarea>
            </div>

            <div class="flex flex-col sm:flex-row gap-4">
              <select v-model="demoModel" class="input-field">
                <option value="t2v-540p">Text-to-Video (540P)</option>
                <option value="t2v-720p">Text-to-Video (720P)</option>
                <option value="i2v-540p">Image-to-Video (540P)</option>
              </select>
              
              <button 
                @click="generateDemo"
                :disabled="isGenerating"
                class="btn-primary flex-shrink-0"
                :class="{ 'opacity-50 cursor-not-allowed': isGenerating }"
              >
                <span v-if="!isGenerating">🎬 Generate Demo</span>
                <span v-else class="flex items-center">
                  <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Generating...
                </span>
              </button>
            </div>

            <!-- Demo Result -->
            <div v-if="demoResult" class="mt-6 p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
              <div class="text-green-400 font-medium">✅ Demo Video Generated!</div>
              <div class="text-white/70 text-sm mt-1">{{ demoResult }}</div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Features Preview -->
    <section class="section-padding">
      <div class="container-max">
        <div class="text-center mb-16">
          <h2 class="text-4xl font-bold text-white mb-4">Powerful Features</h2>
          <p class="text-xl text-white/70">Everything you need for professional video creation</p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          <div v-for="feature in features" :key="feature.name" class="card text-center">
            <div class="text-4xl mb-4">{{ feature.icon }}</div>
            <h3 class="text-xl font-semibold text-white mb-2">{{ feature.name }}</h3>
            <p class="text-white/70">{{ feature.description }}</p>
          </div>
        </div>

        <div class="text-center mt-12">
          <router-link to="/features" class="btn-secondary">
            View All Features
          </router-link>
        </div>
      </div>
    </section>

    <!-- CTA Section -->
    <section class="section-padding bg-gradient-to-r from-capsule-blue/10 to-capsule-purple/10">
      <div class="container-max text-center">
        <h2 class="text-4xl font-bold text-white mb-4">Ready to Create?</h2>
        <p class="text-xl text-white/70 mb-8 max-w-2xl mx-auto">
          Join thousands of creators using Capsule Movie AI to produce stunning videos
        </p>
        <router-link to="/studio" class="btn-primary text-lg px-8 py-4">
          Start Your First Video
        </router-link>
      </div>
    </section>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const demoPrompt = ref('A graceful white swan swimming in a serene lake at dawn, with mist rising from the water.')
const demoModel = ref('t2v-540p')
const isGenerating = ref(false)
const demoResult = ref('')
const demoSection = ref(null)

const features = [
  {
    name: 'Text-to-Video',
    icon: '📝',
    description: 'Transform text descriptions into stunning videos using advanced AI models'
  },
  {
    name: 'Image-to-Video',
    icon: '🖼️',
    description: 'Animate static images with realistic motion and seamless transitions'
  },
  {
    name: 'Infinite Length',
    icon: '🎞️',
    description: 'Create videos of any length using Diffusion Forcing technology'
  },
  {
    name: 'High Quality',
    icon: '✨',
    description: 'Generate videos up to 720P resolution with professional quality'
  },
  {
    name: 'Fast Generation',
    icon: '⚡',
    description: 'Optimized inference with TeaCache and multi-GPU acceleration'
  },
  {
    name: 'Frame Control',
    icon: '🎯',
    description: 'Precise control over start and end frames for perfect results'
  }
]

const scrollToDemo = () => {
  demoSection.value?.scrollIntoView({ behavior: 'smooth' })
}

const generateDemo = async () => {
  if (!demoPrompt.value.trim()) return
  
  isGenerating.value = true
  demoResult.value = ''
  
  try {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 3000))
    demoResult.value = `Generated ${demoModel.value} video: "${demoPrompt.value.substring(0, 50)}..."`
  } catch (error) {
    console.error('Demo generation failed:', error)
  } finally {
    isGenerating.value = false
  }
}
</script>