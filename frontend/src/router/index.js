import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Studio from '../views/Studio.vue'
import Features from '../views/Features.vue'
import Pricing from '../views/Pricing.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: Home
    },
    {
      path: '/studio',
      name: 'studio',
      component: Studio
    },
    {
      path: '/features',
      name: 'features',
      component: Features
    },
    {
      path: '/pricing',
      name: 'pricing',
      component: Pricing
    }
  ]
})

export default router