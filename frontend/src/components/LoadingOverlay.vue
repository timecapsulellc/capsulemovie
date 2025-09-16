<template>
  <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
    <div class="glass-effect rounded-lg p-8 max-w-sm w-full mx-4 text-center">
      <!-- Loading animation -->
      <div class="mb-6">
        <div class="relative w-20 h-20 mx-auto">
          <div class="absolute inset-0 border-4 border-white/20 rounded-full"></div>
          <div class="absolute inset-0 border-4 border-transparent border-t-capsule-blue rounded-full animate-spin"></div>
          <div class="absolute inset-2 border-4 border-transparent border-t-capsule-purple rounded-full animate-spin" style="animation-direction: reverse; animation-duration: 1.5s;"></div>
        </div>
      </div>
      
      <!-- Loading text -->
      <h3 class="text-lg font-semibold text-white mb-2">{{ title }}</h3>
      <p class="text-white/70 text-sm">{{ message }}</p>
      
      <!-- Progress bar -->
      <div v-if="progress !== null" class="mt-4">
        <div class="w-full bg-white/20 rounded-full h-2">
          <div 
            class="bg-gradient-to-r from-capsule-blue to-capsule-purple h-2 rounded-full transition-all duration-300"
            :style="{ width: `${progress}%` }"
          ></div>
        </div>
        <div class="text-white/60 text-xs mt-2">{{ progress }}%</div>
      </div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    default: 'Processing...'
  },
  message: {
    type: String,
    default: 'Please wait while we process your request'
  },
  progress: {
    type: Number,
    default: null
  }
})
</script>