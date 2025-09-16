<template>
  <footer class="bg-gray-900 border-t border-white/10">
    <div class="container-max section-padding">
      <div class="grid grid-cols-1 md:grid-cols-4 gap-8">
        <!-- Brand -->
        <div class="col-span-1 md:col-span-2">
          <div class="flex items-center space-x-3 mb-4">
            <div class="w-8 h-8 bg-gradient-to-r from-capsule-blue to-capsule-purple rounded-lg flex items-center justify-center">
              <span class="text-white font-bold text-lg">C</span>
            </div>
            <span class="text-xl font-bold text-white">Capsule Movie</span>
          </div>
          <p class="text-white/70 mb-4 max-w-md">
            Advanced AI video generation platform powered by Diffusion Forcing technology. 
            Create infinite-length films with professional quality.
          </p>
          <div class="flex space-x-4">
            <a href="#" class="text-white/60 hover:text-white transition-colors">
              <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                <path d="M24 4.557c-.883.392-1.832.656-2.828.775 1.017-.609 1.798-1.574 2.165-2.724-.951.564-2.005.974-3.127 1.195-.897-.957-2.178-1.555-3.594-1.555-3.179 0-5.515 2.966-4.797 6.045-4.091-.205-7.719-2.165-10.148-5.144-1.29 2.213-.669 5.108 1.523 6.574-.806-.026-1.566-.247-2.229-.616-.054 2.281 1.581 4.415 3.949 4.89-.693.188-1.452.232-2.224.084.626 1.956 2.444 3.379 4.6 3.419-2.07 1.623-4.678 2.348-7.29 2.04 2.179 1.397 4.768 2.212 7.548 2.212 9.142 0 14.307-7.721 13.995-14.646.962-.695 1.797-1.562 2.457-2.549z"/>
              </svg>
            </a>
            <a href="#" class="text-white/60 hover:text-white transition-colors">
              <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                <path d="M22.46 6c-.77.35-1.6.58-2.46.69.88-.53 1.56-1.37 1.88-2.38-.83.5-1.75.85-2.72 1.05C18.37 4.5 17.26 4 16 4c-2.35 0-4.27 1.92-4.27 4.29 0 .34.04.67.11.98C8.28 9.09 5.11 7.38 3 4.79c-.37.63-.58 1.37-.58 2.15 0 1.49.75 2.81 1.91 3.56-.71 0-1.37-.2-1.95-.5v.03c0 2.08 1.48 3.82 3.44 4.21a4.22 4.22 0 0 1-1.93.07 4.28 4.28 0 0 0 4 2.98 8.521 8.521 0 0 1-5.33 1.84c-.34 0-.68-.02-1.02-.06C3.44 20.29 5.7 21 8.12 21 16 21 20.33 14.46 20.33 8.79c0-.19 0-.37-.01-.56.84-.6 1.56-1.36 2.14-2.23z"/>
              </svg>
            </a>
            <a href="#" class="text-white/60 hover:text-white transition-colors">
              <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12.017 0C5.396 0 .029 5.367.029 11.987c0 5.079 3.158 9.417 7.618 11.174-.105-.949-.199-2.403.041-3.439.219-.937 1.219-5.160 1.219-5.160s-.312-.623-.312-1.543c0-1.445.839-2.524 1.895-2.524.893 0 1.326.669 1.326 1.469 0 .896-.57 2.237-.864 3.475-.246 1.064.533 1.931 1.588 1.931 1.902 0 3.362-2.003 3.362-4.893 0-2.559-1.838-4.351-4.465-4.351-3.043 0-4.83 2.281-4.83 4.64 0 .919.354 1.903.794 2.437.087.105.099.198.073.304-.08.331-.256 1.045-.293 1.191-.048.191-.157.232-.364.141-1.298-.604-2.107-2.502-2.107-4.026 0-3.372 2.45-6.466 7.065-6.466 3.71 0 6.597 2.643 6.597 6.181 0 3.688-2.325 6.653-5.552 6.653-1.085 0-2.104-.564-2.45-1.231 0 0-.538 2.049-.669 2.549-.242.93-.896 2.093-1.332 2.807.999.308 2.058.47 3.17.47 6.624 0 11.99-5.367 11.99-11.988C24.007 5.367 18.641.001 12.017.001z"/>
              </svg>
            </a>
          </div>
        </div>

        <!-- Links -->
        <div>
          <h3 class="text-white font-semibold mb-4">Product</h3>
          <ul class="space-y-2">
            <li><router-link to="/features" class="text-white/60 hover:text-white transition-colors">Features</router-link></li>
            <li><router-link to="/pricing" class="text-white/60 hover:text-white transition-colors">Pricing</router-link></li>
            <li><router-link to="/studio" class="text-white/60 hover:text-white transition-colors">Studio</router-link></li>
            <li><a href="#" class="text-white/60 hover:text-white transition-colors">API</a></li>
          </ul>
        </div>

        <div>
          <h3 class="text-white font-semibold mb-4">Company</h3>
          <ul class="space-y-2">
            <li><a href="#" class="text-white/60 hover:text-white transition-colors">About</a></li>
            <li><a href="#" class="text-white/60 hover:text-white transition-colors">Blog</a></li>
            <li><a href="#" class="text-white/60 hover:text-white transition-colors">Careers</a></li>
            <li><a href="#" class="text-white/60 hover:text-white transition-colors">Contact</a></li>
          </ul>
        </div>
      </div>

      <!-- Bottom bar -->
      <div class="border-t border-white/10 mt-8 pt-8 flex flex-col md:flex-row justify-between items-center">
        <div class="text-white/60 text-sm">
          © 2025 Capsule Movie AI. Inspired by SkyReels technology.
        </div>
        <div class="flex space-x-6 mt-4 md:mt-0">
          <a href="#" class="text-white/60 hover:text-white text-sm transition-colors">Privacy Policy</a>
          <a href="#" class="text-white/60 hover:text-white text-sm transition-colors">Terms of Service</a>
          <a href="#" class="text-white/60 hover:text-white text-sm transition-colors">Support</a>
        </div>
      </div>
    </div>
  </footer>
</template>

<script setup>
// Footer component - no logic needed
</script>