<template>
  <div id="app" class="min-h-screen bg-capsule-dark">
    <!-- Navigation -->
    <Navigation />
    
    <!-- Main Content -->
    <main class="flex-1">
      <router-view v-slot="{ Component }">
        <Transition name="page" mode="out-in">
          <component :is="Component" />
        </Transition>
      </router-view>
    </main>
    
    <!-- Footer -->
    <Footer />
    
    <!-- Global Loading Overlay -->
    <LoadingOverlay v-if="isLoading" />
  </div>
</template>

<script setup>
import { ref, provide } from 'vue'
import Navigation from './components/Navigation.vue'
import Footer from './components/Footer.vue'
import LoadingOverlay from './components/LoadingOverlay.vue'

const isLoading = ref(false)

// Global loading state management
const setLoading = (state) => {
  isLoading.value = state
}

// Provide loading function to child components
provide('setLoading', setLoading)
</script>

<style scoped>
/* Page transitions */
.page-enter-active,
.page-leave-active {
  transition: all 0.3s ease;
}

.page-enter-from {
  opacity: 0;
  transform: translateY(20px);
}

.page-leave-to {
  opacity: 0;
  transform: translateY(-20px);
}
</style>