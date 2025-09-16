<template>
  <nav class="fixed top-0 w-full z-50 glass-effect">
    <div class="container-max">
      <div class="flex items-center justify-between h-16 px-4 sm:px-6 lg:px-8">
        <!-- Logo -->
        <div class="flex items-center">
          <router-link to="/" class="flex items-center space-x-3">
            <div class="w-8 h-8 bg-gradient-to-r from-capsule-blue to-capsule-purple rounded-lg flex items-center justify-center">
              <span class="text-white font-bold text-lg">C</span>
            </div>
            <span class="text-xl font-bold text-white">Capsule Movie</span>
          </router-link>
        </div>

        <!-- Desktop Navigation -->
        <div class="hidden md:block">
          <div class="ml-10 flex items-baseline space-x-8">
            <router-link
              v-for="item in navigation"
              :key="item.name"
              :to="item.href"
              class="text-white/80 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200"
              :class="{ 'text-white bg-white/10': $route.path === item.href }"
            >
              {{ item.name }}
            </router-link>
          </div>
        </div>

        <!-- Right side buttons -->
        <div class="hidden md:flex items-center space-x-4">
          <button 
            @click="openAuthModal('login')"
            class="btn-ghost"
          >
            Sign In
          </button>
          <button 
            @click="openAuthModal('signup')"
            class="btn-primary"
          >
            Get Started
          </button>
        </div>

        <!-- Mobile menu button -->
        <div class="md:hidden">
          <button
            @click="mobileMenuOpen = !mobileMenuOpen"
            class="text-white hover:text-white/80 focus:outline-none"
          >
            <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path v-if="!mobileMenuOpen" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
              <path v-else stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>
    </div>

    <!-- Mobile Navigation -->
    <Transition name="mobile-menu">
      <div v-if="mobileMenuOpen" class="md:hidden glass-effect border-t border-white/10">
        <div class="px-2 pt-2 pb-3 space-y-1">
          <router-link
            v-for="item in navigation"
            :key="item.name"
            :to="item.href"
            @click="mobileMenuOpen = false"
            class="text-white/80 hover:text-white block px-3 py-2 rounded-md text-base font-medium transition-colors duration-200"
          >
            {{ item.name }}
          </router-link>
          <div class="pt-4 pb-3 border-t border-white/10">
            <div class="flex items-center space-x-3">
              <button 
                @click="openAuthModal('login')"
                class="btn-ghost w-full"
              >
                Sign In
              </button>
              <button 
                @click="openAuthModal('signup')"
                class="btn-primary w-full"
              >
                Get Started
              </button>
            </div>
          </div>
        </div>
      </div>
    </Transition>

    <!-- Auth Modal -->
    <AuthModal 
      v-if="authModalOpen"
      :mode="authMode"
      @close="closeAuthModal"
    />
  </nav>
</template>

<script setup>
import { ref } from 'vue'
import AuthModal from './AuthModal.vue'

const mobileMenuOpen = ref(false)
const authModalOpen = ref(false)
const authMode = ref('login')

const navigation = [
  { name: 'Home', href: '/' },
  { name: 'Studio', href: '/studio' },
  { name: 'Features', href: '/features' },
  { name: 'Pricing', href: '/pricing' },
]

const openAuthModal = (mode) => {
  authMode.value = mode
  authModalOpen.value = true
  mobileMenuOpen.value = false
}

const closeAuthModal = () => {
  authModalOpen.value = false
}
</script>

<style scoped>
.mobile-menu-enter-active,
.mobile-menu-leave-active {
  transition: all 0.3s ease;
}

.mobile-menu-enter-from,
.mobile-menu-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}
</style>