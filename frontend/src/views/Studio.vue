<template>
  <div class="min-h-screen pt-16">
    <!-- Header -->
    <div class="section-padding bg-gradient-to-r from-capsule-dark to-gray-900">
      <div class="container-max">
        <div class="text-center">
          <h1 class="text-4xl font-bold text-white mb-4">AI Video Studio</h1>
          <p class="text-xl text-white/70 max-w-2xl mx-auto">
            Create professional videos with advanced AI technology. Choose your generation method below.
          </p>
        </div>
      </div>
    </div>

    <!-- Generation Mode Selector -->
    <div class="section-padding">
      <div class="container-max">
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <button
            v-for="mode in generationModes"
            :key="mode.id"
            @click="selectedMode = mode.id"
            class="card text-center p-8 transition-all duration-300 hover:scale-105"
            :class="{ 'ring-2 ring-capsule-blue bg-capsule-blue/10': selectedMode === mode.id }"
          >
            <div class="text-4xl mb-4">{{ mode.icon }}</div>
            <h3 class="text-xl font-semibold text-white mb-2">{{ mode.name }}</h3>
            <p class="text-white/70 text-sm">{{ mode.description }}</p>
          </button>
        </div>

        <!-- Generation Interface -->
        <div class="max-w-4xl mx-auto">
          <!-- Text-to-Video Interface -->
          <div v-if="selectedMode === 'text2video'" class="card">
            <div class="mb-6">
              <h2 class="text-2xl font-bold text-white mb-2">Text-to-Video Generation</h2>
              <p class="text-white/70">Describe your video and let AI bring it to life</p>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <!-- Controls -->
              <div class="space-y-6">
                <div>
                  <label class="block text-sm font-medium text-white/80 mb-2">Video Description</label>
                  <textarea
                    v-model="textPrompt"
                    class="input-field h-32 resize-none"
                    placeholder="A serene lake surrounded by towering mountains, with swans gracefully gliding across the water..."
                  ></textarea>
                </div>

                <div class="grid grid-cols-2 gap-4">
                  <div>
                    <label class="block text-sm font-medium text-white/80 mb-2">Model</label>
                    <select v-model="textModel" class="input-field">
                      <option value="540p">SkyReels-T2V-540P</option>
                      <option value="720p">SkyReels-T2V-720P</option>
                    </select>
                  </div>
                  <div>
                    <label class="block text-sm font-medium text-white/80 mb-2">Duration</label>
                    <select v-model="textDuration" class="input-field">
                      <option value="4s">4 seconds (97 frames)</option>
                      <option value="10s">10 seconds (257 frames)</option>
                      <option value="30s">30 seconds (737 frames)</option>
                    </select>
                  </div>
                </div>

                <div class="grid grid-cols-2 gap-4">
                  <div>
                    <label class="block text-sm font-medium text-white/80 mb-2">Guidance Scale: {{ textGuidance }}</label>
                    <input
                      v-model.number="textGuidance"
                      type="range"
                      min="1"
                      max="15"
                      step="0.5"
                      class="slider w-full"
                    />
                  </div>
                  <div>
                    <label class="block text-sm font-medium text-white/80 mb-2">Quality Steps: {{ textSteps }}</label>
                    <input
                      v-model.number="textSteps"
                      type="range"
                      min="10"
                      max="100"
                      step="5"
                      class="slider w-full"
                    />
                  </div>
                </div>

                <div class="flex items-center space-x-4">
                  <label class="flex items-center space-x-2">
                    <input v-model="useEnhancer" type="checkbox" class="w-4 h-4 text-capsule-blue bg-transparent border-white/20 rounded">
                    <span class="text-white/80 text-sm">Use Prompt Enhancer</span>
                  </label>
                </div>

                <button
                  @click="generateVideo('text2video')"
                  :disabled="isGenerating || !textPrompt.trim()"
                  class="w-full btn-primary py-4"
                  :class="{ 'opacity-50 cursor-not-allowed': isGenerating || !textPrompt.trim() }"
                >
                  <span v-if="!isGenerating">🎬 Generate Video</span>
                  <span v-else class="flex items-center justify-center">
                    <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Generating... {{ generationProgress }}%
                  </span>
                </button>
              </div>

              <!-- Preview -->
              <div>
                <label class="block text-sm font-medium text-white/80 mb-2">Video Preview</label>
                <div class="aspect-video bg-gradient-to-br from-capsule-blue/20 to-capsule-purple/20 rounded-lg flex items-center justify-center">
                  <div v-if="!generatedVideo" class="text-center text-white/60">
                    <div class="w-16 h-16 mx-auto mb-4 bg-white/10 rounded-full flex items-center justify-center">
                      🎬
                    </div>
                    <p>Generated video will appear here</p>
                  </div>
                  <div v-else class="text-center text-white">
                    <div class="w-16 h-16 mx-auto mb-4 bg-green-500/20 rounded-full flex items-center justify-center">
                      ✅
                    </div>
                    <p class="font-medium">Video Generated Successfully!</p>
                    <p class="text-sm text-white/70 mt-1">{{ generatedVideo }}</p>
                  </div>
                </div>

                <!-- Generation Progress -->
                <div v-if="isGenerating" class="mt-4">
                  <div class="w-full bg-white/20 rounded-full h-2">
                    <div 
                      class="bg-gradient-to-r from-capsule-blue to-capsule-purple h-2 rounded-full transition-all duration-300"
                      :style="{ width: `${generationProgress}%` }"
                    ></div>
                  </div>
                  <div class="text-white/60 text-xs mt-2 text-center">{{ generationStatus }}</div>
                </div>
              </div>
            </div>
          </div>

          <!-- Image-to-Video Interface -->
          <div v-else-if="selectedMode === 'image2video'" class="card">
            <div class="mb-6">
              <h2 class="text-2xl font-bold text-white mb-2">Image-to-Video Generation</h2>
              <p class="text-white/70">Upload an image and animate it with AI</p>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <!-- Controls -->
              <div class="space-y-6">
                <div>
                  <label class="block text-sm font-medium text-white/80 mb-2">Upload Image</label>
                  <div class="border-2 border-dashed border-white/20 rounded-lg p-8 text-center hover:border-capsule-blue/50 transition-colors cursor-pointer">
                    <div class="text-white/60">
                      <div class="w-12 h-12 mx-auto mb-4 bg-white/10 rounded-full flex items-center justify-center">
                        📸
                      </div>
                      <p>Click to upload or drag and drop</p>
                      <p class="text-xs mt-1">PNG, JPG, WEBP up to 10MB</p>
                    </div>
                  </div>
                </div>

                <div>
                  <label class="block text-sm font-medium text-white/80 mb-2">Animation Description</label>
                  <textarea
                    v-model="imagePrompt"
                    class="input-field h-24 resize-none"
                    placeholder="Describe how the image should move and animate..."
                  ></textarea>
                </div>

                <div class="grid grid-cols-2 gap-4">
                  <div>
                    <label class="block text-sm font-medium text-white/80 mb-2">Model</label>
                    <select v-model="imageModel" class="input-field">
                      <option value="540p">SkyReels-I2V-540P</option>
                      <option value="720p">SkyReels-I2V-720P</option>
                    </select>
                  </div>
                  <div>
                    <label class="block text-sm font-medium text-white/80 mb-2">Duration</label>
                    <select v-model="imageDuration" class="input-field">
                      <option value="4s">4 seconds</option>
                      <option value="8s">8 seconds</option>
                    </select>
                  </div>
                </div>

                <button
                  @click="generateVideo('image2video')"
                  :disabled="isGenerating"
                  class="w-full btn-primary py-4"
                  :class="{ 'opacity-50 cursor-not-allowed': isGenerating }"
                >
                  🎬 Animate Image
                </button>
              </div>

              <!-- Preview -->
              <div>
                <label class="block text-sm font-medium text-white/80 mb-2">Animation Preview</label>
                <div class="aspect-video bg-gradient-to-br from-capsule-purple/20 to-capsule-gold/20 rounded-lg flex items-center justify-center">
                  <div class="text-center text-white/60">
                    <div class="w-16 h-16 mx-auto mb-4 bg-white/10 rounded-full flex items-center justify-center">
                      🖼️
                    </div>
                    <p>Animated video will appear here</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Diffusion Forcing Interface -->
          <div v-else-if="selectedMode === 'longform'" class="card">
            <div class="mb-6">
              <h2 class="text-2xl font-bold text-white mb-2">Long-Form Video Generation</h2>
              <p class="text-white/70">Create infinite-length videos using Diffusion Forcing</p>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <!-- Controls -->
              <div class="space-y-6">
                <div>
                  <label class="block text-sm font-medium text-white/80 mb-2">Video Concept</label>
                  <textarea
                    v-model="longPrompt"
                    class="input-field h-32 resize-none"
                    placeholder="A graceful white swan swimming in a serene lake at dawn, creating ripples in the perfectly still water..."
                  ></textarea>
                </div>

                <div class="grid grid-cols-2 gap-4">
                  <div>
                    <label class="block text-sm font-medium text-white/80 mb-2">Length</label>
                    <select v-model="longDuration" class="input-field">
                      <option value="10s">10 seconds</option>
                      <option value="30s">30 seconds</option>
                      <option value="60s">60 seconds</option>
                      <option value="custom">Custom</option>
                    </select>
                  </div>
                  <div>
                    <label class="block text-sm font-medium text-white/80 mb-2">Quality</label>
                    <select v-model="longQuality" class="input-field">
                      <option value="540p">540P (Fast)</option>
                      <option value="720p">720P (High Quality)</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label class="block text-sm font-medium text-white/80 mb-2">Advanced Settings</label>
                  <div class="space-y-3">
                    <label class="flex items-center space-x-2">
                      <input v-model="useAsyncGeneration" type="checkbox" class="w-4 h-4 text-capsule-blue bg-transparent border-white/20 rounded">
                      <span class="text-white/80 text-sm">Asynchronous Generation (Better for long videos)</span>
                    </label>
                    <label class="flex items-center space-x-2">
                      <input v-model="useFrameControl" type="checkbox" class="w-4 h-4 text-capsule-blue bg-transparent border-white/20 rounded">
                      <span class="text-white/80 text-sm">Enable Start/End Frame Control</span>
                    </label>
                  </div>
                </div>

                <button
                  @click="generateVideo('longform')"
                  :disabled="isGenerating || !longPrompt.trim()"
                  class="w-full btn-primary py-4"
                  :class="{ 'opacity-50 cursor-not-allowed': isGenerating || !longPrompt.trim() }"
                >
                  <span v-if="!isGenerating">🎞️ Generate Long Video</span>
                  <span v-else class="flex items-center justify-center">
                    <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Generating... {{ generationProgress }}%
                  </span>
                </button>
              </div>

              <!-- Preview -->
              <div>
                <label class="block text-sm font-medium text-white/80 mb-2">Long Video Preview</label>
                <div class="aspect-video bg-gradient-to-br from-capsule-gold/20 to-capsule-blue/20 rounded-lg flex items-center justify-center">
                  <div class="text-center text-white/60">
                    <div class="w-16 h-16 mx-auto mb-4 bg-white/10 rounded-full flex items-center justify-center">
                      🎞️
                    </div>
                    <p>Long-form video will appear here</p>
                    <p class="text-xs mt-1">Supports infinite length generation</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Recent Generations -->
        <div class="mt-16">
          <h3 class="text-2xl font-bold text-white mb-8 text-center">Recent Generations</h3>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div v-for="video in recentVideos" :key="video.id" class="card">
              <div class="aspect-video bg-gradient-to-br from-gray-700 to-gray-800 rounded-lg mb-4 flex items-center justify-center">
                <div class="text-white/60 text-center">
                  <div class="w-12 h-12 mx-auto mb-2 bg-white/10 rounded-full flex items-center justify-center">
                    ▶️
                  </div>
                  <p class="text-xs">{{ video.type }}</p>
                </div>
              </div>
              <h4 class="text-white font-medium mb-1">{{ video.title }}</h4>
              <p class="text-white/60 text-sm">{{ video.duration }} • {{ video.resolution }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const selectedMode = ref('text2video')
const isGenerating = ref(false)
const generationProgress = ref(0)
const generationStatus = ref('')
const generatedVideo = ref('')

// Text-to-Video state
const textPrompt = ref('A serene lake surrounded by towering mountains, with swans gracefully gliding across the water and sunlight dancing on the surface.')
const textModel = ref('540p')
const textDuration = ref('4s')
const textGuidance = ref(6.0)
const textSteps = ref(30)
const useEnhancer = ref(false)

// Image-to-Video state
const imagePrompt = ref('The image comes to life with gentle movements and natural motion.')
const imageModel = ref('540p')
const imageDuration = ref('4s')

// Long-form state
const longPrompt = ref('A graceful white swan swimming in a serene lake at dawn, creating ripples in the perfectly still water.')
const longDuration = ref('10s')
const longQuality = ref('540p')
const useAsyncGeneration = ref(false)
const useFrameControl = ref(false)

const generationModes = [
  {
    id: 'text2video',
    name: 'Text-to-Video',
    icon: '📝',
    description: 'Generate videos from text descriptions using advanced AI models'
  },
  {
    id: 'image2video',
    name: 'Image-to-Video',
    icon: '🖼️',
    description: 'Animate static images with realistic motion and transitions'
  },
  {
    id: 'longform',
    name: 'Long-Form Video',
    icon: '🎞️',
    description: 'Create infinite-length videos using Diffusion Forcing technology'
  }
]

const recentVideos = [
  { id: 1, title: 'Swan Lake Animation', duration: '10s', resolution: '720P', type: 'Text-to-Video' },
  { id: 2, title: 'Mountain Landscape', duration: '4s', resolution: '540P', type: 'Image-to-Video' },
  { id: 3, title: 'Ocean Waves', duration: '30s', resolution: '720P', type: 'Long-Form' },
]

const generateVideo = async (type) => {
  isGenerating.value = true
  generationProgress.value = 0
  generatedVideo.value = ''

  const statuses = [
    'Initializing AI model...',
    'Processing input...',
    'Generating frames...',
    'Applying diffusion...',
    'Optimizing quality...',
    'Finalizing video...'
  ]

  try {
    for (let i = 0; i < statuses.length; i++) {
      generationStatus.value = statuses[i]
      
      // Simulate progress
      for (let progress = i * 16; progress < (i + 1) * 16; progress++) {
        generationProgress.value = Math.min(progress, 100)
        await new Promise(resolve => setTimeout(resolve, 100))
      }
    }

    generationProgress.value = 100
    generationStatus.value = 'Complete!'
    
    // Set result based on type
    const prompt = type === 'text2video' ? textPrompt.value : 
                  type === 'image2video' ? imagePrompt.value : longPrompt.value
    
    generatedVideo.value = `${type} generated successfully: "${prompt.substring(0, 50)}..."`
    
  } catch (error) {
    console.error('Generation failed:', error)
    generationStatus.value = 'Generation failed'
  } finally {
    setTimeout(() => {
      isGenerating.value = false
    }, 1000)
  }
}
</script>

<style scoped>
.slider {
  -webkit-appearance: none;
  appearance: none;
  height: 6px;
  border-radius: 3px;
  background: rgba(255, 255, 255, 0.2);
  outline: none;
}

.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: linear-gradient(135deg, #3b82f6, #8b5cf6);
  cursor: pointer;
}

.slider::-moz-range-thumb {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: linear-gradient(135deg, #3b82f6, #8b5cf6);
  cursor: pointer;
  border: none;
}
</style>