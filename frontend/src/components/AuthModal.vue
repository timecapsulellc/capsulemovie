<template>
  <div class="fixed inset-0 z-50 overflow-y-auto" aria-labelledby="modal-title" role="dialog" aria-modal="true">
    <!-- Background overlay -->
    <div class="flex items-end justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
      <div class="fixed inset-0 bg-black/50 backdrop-blur-sm transition-opacity" @click="$emit('close')"></div>

      <!-- Modal panel -->
      <div class="inline-block align-bottom glass-effect rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full">
        <!-- Header -->
        <div class="px-6 py-4 border-b border-white/10">
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-3">
              <div class="w-8 h-8 bg-gradient-to-r from-capsule-blue to-capsule-purple rounded-lg flex items-center justify-center">
                <span class="text-white font-bold">C</span>
              </div>
              <h3 class="text-lg font-medium text-white">Capsule Movie</h3>
            </div>
            <button @click="$emit('close')" class="text-white/60 hover:text-white">
              <svg class="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <!-- Content -->
        <div class="px-6 py-6">
          <div class="text-center mb-6">
            <h2 class="text-2xl font-bold text-white mb-2">
              {{ mode === 'login' ? 'Welcome Back' : 'Join Capsule Movie' }}
            </h2>
            <p class="text-white/70">
              {{ mode === 'login' ? 'Sign in to your account' : 'Create your account to start generating videos' }}
            </p>
          </div>

          <!-- Social Login Buttons -->
          <div class="space-y-3 mb-6">
            <button class="w-full flex items-center justify-center px-4 py-3 border border-white/20 rounded-lg text-white hover:bg-white/5 transition-colors">
              <svg class="w-5 h-5 mr-3" viewBox="0 0 24 24">
                <path fill="currentColor" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="currentColor" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="currentColor" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="currentColor" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              Continue with Google
            </button>
            
            <button class="w-full flex items-center justify-center px-4 py-3 border border-white/20 rounded-lg text-white hover:bg-white/5 transition-colors">
              <svg class="w-5 h-5 mr-3" fill="currentColor" viewBox="0 0 24 24">
                <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/>
              </svg>
              Continue with Facebook
            </button>
            
            <button class="w-full flex items-center justify-center px-4 py-3 border border-white/20 rounded-lg text-white hover:bg-white/5 transition-colors">
              <svg class="w-5 h-5 mr-3" fill="currentColor" viewBox="0 0 24 24">
                <path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z"/>
              </svg>
              Continue with X
            </button>
          </div>

          <!-- Divider -->
          <div class="relative mb-6">
            <div class="absolute inset-0 flex items-center">
              <div class="w-full border-t border-white/20"></div>
            </div>
            <div class="relative flex justify-center text-sm">
              <span class="px-2 bg-capsule-dark text-white/60">OR</span>
            </div>
          </div>

          <!-- Email Form -->
          <form @submit.prevent="handleSubmit" class="space-y-4">
            <div>
              <label class="block text-sm font-medium text-white/80 mb-2">Email</label>
              <input
                v-model="email"
                type="email"
                required
                class="input-field"
                placeholder="Enter your email"
              />
            </div>

            <div v-if="mode === 'signup'">
              <label class="block text-sm font-medium text-white/80 mb-2">Full Name</label>
              <input
                v-model="fullName"
                type="text"
                required
                class="input-field"
                placeholder="Enter your full name"
              />
            </div>

            <div>
              <label class="block text-sm font-medium text-white/80 mb-2">Password</label>
              <input
                v-model="password"
                type="password"
                required
                class="input-field"
                placeholder="Enter your password"
              />
            </div>

            <div v-if="mode === 'signup'" class="flex items-start">
              <div class="flex items-center h-5">
                <input
                  v-model="agreeTerms"
                  type="checkbox"
                  required
                  class="w-4 h-4 text-capsule-blue bg-transparent border-white/20 rounded focus:ring-capsule-blue focus:ring-2"
                />
              </div>
              <div class="ml-3 text-sm">
                <label class="text-white/70">
                  I agree to the 
                  <a href="#" class="text-capsule-blue hover:text-capsule-purple">Terms of Service</a>
                  and 
                  <a href="#" class="text-capsule-blue hover:text-capsule-purple">Privacy Policy</a>
                </label>
              </div>
            </div>

            <button
              type="submit"
              :disabled="isSubmitting"
              class="w-full btn-primary"
              :class="{ 'opacity-50 cursor-not-allowed': isSubmitting }"
            >
              <span v-if="!isSubmitting">
                {{ mode === 'login' ? 'Sign In' : 'Create Account' }}
              </span>
              <span v-else class="flex items-center justify-center">
                <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                {{ mode === 'login' ? 'Signing In...' : 'Creating Account...' }}
              </span>
            </button>
          </form>

          <!-- Switch Mode -->
          <div class="mt-6 text-center">
            <button
              @click="toggleMode"
              class="text-sm text-white/70 hover:text-white"
            >
              {{ mode === 'login' ? "Don't have an account? Sign up" : "Already have an account? Sign in" }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  mode: {
    type: String,
    default: 'login'
  }
})

const emit = defineEmits(['close'])

const email = ref('')
const password = ref('')
const fullName = ref('')
const agreeTerms = ref(false)
const isSubmitting = ref(false)

const toggleMode = () => {
  const newMode = props.mode === 'login' ? 'signup' : 'login'
  emit('close')
  // Re-open with new mode (parent component handles this)
}

const handleSubmit = async () => {
  isSubmitting.value = true
  
  try {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000))
    
    // Handle success
    console.log('Auth success:', { mode: props.mode, email: email.value })
    emit('close')
  } catch (error) {
    console.error('Auth error:', error)
  } finally {
    isSubmitting.value = false
  }
}
</script>