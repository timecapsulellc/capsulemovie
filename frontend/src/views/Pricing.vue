<template>
  <div class="min-h-screen pt-16">
    <!-- Header -->
    <div class="section-padding bg-gradient-to-r from-capsule-dark to-gray-900">
      <div class="container-max text-center">
        <h1 class="text-4xl font-bold text-white mb-4">Simple, Transparent Pricing</h1>
        <p class="text-xl text-white/70 max-w-3xl mx-auto">
          Choose the plan that fits your creative needs. All plans include access to our cutting-edge AI models.
        </p>
      </div>
    </div>

    <!-- Pricing Cards -->
    <div class="section-padding">
      <div class="container-max">
        <div class="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
          <!-- Starter Plan -->
          <div class="card text-center">
            <div class="mb-6">
              <h3 class="text-xl font-semibold text-white mb-2">Starter</h3>
              <div class="text-3xl font-bold text-white mb-1">$9</div>
              <div class="text-white/60 text-sm">per month</div>
            </div>

            <ul class="space-y-3 mb-8 text-left">
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                10 videos per month
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                Up to 540P resolution
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                4-second videos
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                Text-to-Video generation
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                Community support
              </li>
            </ul>

            <button class="w-full btn-secondary">Get Started</button>
          </div>

          <!-- Professional Plan -->
          <div class="card text-center ring-2 ring-capsule-blue relative">
            <div class="absolute -top-3 left-1/2 transform -translate-x-1/2">
              <span class="bg-gradient-to-r from-capsule-blue to-capsule-purple text-white text-xs font-medium px-3 py-1 rounded-full">
                Most Popular
              </span>
            </div>

            <div class="mb-6 mt-4">
              <h3 class="text-xl font-semibold text-white mb-2">Professional</h3>
              <div class="text-3xl font-bold text-white mb-1">$29</div>
              <div class="text-white/60 text-sm">per month</div>
            </div>

            <ul class="space-y-3 mb-8 text-left">
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                100 videos per month
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                Up to 720P resolution
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                Up to 30-second videos
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                All generation modes
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                Priority processing
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                Email support
              </li>
            </ul>

            <button class="w-full btn-primary">Start Free Trial</button>
          </div>

          <!-- Enterprise Plan -->
          <div class="card text-center">
            <div class="mb-6">
              <h3 class="text-xl font-semibold text-white mb-2">Enterprise</h3>
              <div class="text-3xl font-bold text-white mb-1">$99</div>
              <div class="text-white/60 text-sm">per month</div>
            </div>

            <ul class="space-y-3 mb-8 text-left">
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                Unlimited videos
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                Full 720P resolution
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                Infinite-length videos
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                API access
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                Custom models
              </li>
              <li class="flex items-center text-white/80">
                <span class="w-2 h-2 bg-capsule-blue rounded-full mr-3"></span>
                Dedicated support
              </li>
            </ul>

            <button class="w-full btn-secondary">Contact Sales</button>
          </div>
        </div>

        <!-- FAQ Section -->
        <div class="mt-20 max-w-4xl mx-auto">
          <h2 class="text-3xl font-bold text-white text-center mb-12">Frequently Asked Questions</h2>
          
          <div class="space-y-6">
            <div v-for="faq in faqs" :key="faq.question" class="card">
              <button
                @click="faq.open = !faq.open"
                class="w-full text-left flex items-center justify-between p-2"
              >
                <h3 class="text-lg font-medium text-white">{{ faq.question }}</h3>
                <svg
                  class="w-5 h-5 text-white/60 transition-transform duration-200"
                  :class="{ 'transform rotate-180': faq.open }"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              <Transition name="fade">
                <div v-if="faq.open" class="mt-4 text-white/70">
                  {{ faq.answer }}
                </div>
              </Transition>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- CTA Section -->
    <div class="section-padding bg-gradient-to-r from-capsule-blue/10 to-capsule-purple/10">
      <div class="container-max text-center">
        <h2 class="text-4xl font-bold text-white mb-4">Ready to Get Started?</h2>
        <p class="text-xl text-white/70 mb-8">
          Start your free trial today and experience the future of video creation
        </p>
        <div class="flex flex-col sm:flex-row items-center justify-center gap-4">
          <button class="btn-primary text-lg px-8 py-4">Start Free Trial</button>
          <button class="btn-ghost text-lg px-8 py-4">Schedule Demo</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { reactive } from 'vue'

const faqs = reactive([
  {
    question: 'What video formats are supported?',
    answer: 'We generate videos in MP4 format with H.264 encoding, compatible with all major platforms and devices.',
    open: false
  },
  {
    question: 'How long does video generation take?',
    answer: 'Generation time varies by video length and quality. Typical 4-second videos take 2-5 minutes, while longer videos may take 10-30 minutes.',
    open: false
  },
  {
    question: 'Can I use generated videos commercially?',
    answer: 'Yes! All videos generated with Professional and Enterprise plans can be used for commercial purposes without attribution.',
    open: false
  },
  {
    question: 'What makes your AI different?',
    answer: 'We use advanced Diffusion Forcing technology that enables infinite-length video generation with superior temporal consistency compared to other platforms.',
    open: false
  },
  {
    question: 'Do you offer API access?',
    answer: 'API access is available with Enterprise plans. Contact our sales team to discuss integration options and custom solutions.',
    open: false
  }
])
</script>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>