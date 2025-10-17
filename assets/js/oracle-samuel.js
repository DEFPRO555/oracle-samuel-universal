/* Oracle Samuel - GitHub Pages JavaScript */
/* © 2025 Dowek Analytics Ltd. All Rights Reserved. */

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// Main initialization function
function initializeApp() {
    setupSmoothScrolling();
    setupNavbarScroll();
    setupAnimations();
    setupLoadingStates();
    setupInteractiveElements();
    setupStreamlitIntegration();
    setupPerformanceOptimizations();
}

// Smooth scrolling for navigation links
function setupSmoothScrolling() {
    const navLinks = document.querySelectorAll('a[href^="#"]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                const offsetTop = targetElement.offsetTop - 80; // Account for fixed navbar
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
                
                // Close mobile menu if open
                const navbarCollapse = document.querySelector('.navbar-collapse');
                if (navbarCollapse.classList.contains('show')) {
                    const navbarToggler = document.querySelector('.navbar-toggler');
                    navbarToggler.click();
                }
            }
        });
    });
}

// Navbar scroll effects
function setupNavbarScroll() {
    const navbar = document.querySelector('.navbar');
    let lastScrollTop = 0;
    
    window.addEventListener('scroll', function() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        // Add/remove scrolled class
        if (scrollTop > 50) {
            navbar.classList.add('navbar-scrolled');
        } else {
            navbar.classList.remove('navbar-scrolled');
        }
        
        // Hide/show navbar on scroll (optional)
        if (scrollTop > lastScrollTop && scrollTop > 100) {
            navbar.style.transform = 'translateY(-100%)';
        } else {
            navbar.style.transform = 'translateY(0)';
        }
        
        lastScrollTop = scrollTop;
    });
}

// Intersection Observer for animations
function setupAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    const animatedElements = document.querySelectorAll('.feature-card, .market-card, .doc-card, .launcher-card');
    animatedElements.forEach(el => {
        observer.observe(el);
    });
}

// Loading states for buttons
function setupLoadingStates() {
    const buttons = document.querySelectorAll('.btn');
    
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            if (this.classList.contains('btn-loading')) return;
            
            const originalText = this.innerHTML;
            this.classList.add('btn-loading');
            this.innerHTML = '<span class="loading"></span> Loading...';
            
            // Remove loading state after 2 seconds (adjust as needed)
            setTimeout(() => {
                this.classList.remove('btn-loading');
                this.innerHTML = originalText;
            }, 2000);
        });
    });
}

// Interactive elements
function setupInteractiveElements() {
    // Parallax effect for hero section
    window.addEventListener('scroll', function() {
        const scrolled = window.pageYOffset;
        const parallax = document.querySelector('.hero-bg-animation');
        
        if (parallax) {
            const speed = scrolled * 0.5;
            parallax.style.transform = `translateY(${speed}px)`;
        }
    });
    
    // Card hover effects
    const cards = document.querySelectorAll('.feature-card, .market-card, .doc-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
    
    // Social links animation
    const socialLinks = document.querySelectorAll('.social-links a');
    socialLinks.forEach(link => {
        link.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px) scale(1.1)';
        });
        
        link.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
}

// Streamlit integration
function setupStreamlitIntegration() {
    // Check if Streamlit app is available
    window.checkStreamlitStatus = async function() {
        try {
            // Replace with your actual Streamlit app URL
            const streamlitUrl = 'https://oracle-samuel-app.streamlit.app/';
            const response = await fetch(streamlitUrl, { method: 'HEAD' });
            return response.ok;
        } catch (error) {
            console.log('Streamlit app not available:', error);
            return false;
        }
    };
    
    // Launch Streamlit app with status check
    window.launchStreamlitApp = async function() {
        const button = event.target.closest('button');
        const originalText = button.innerHTML;
        
        button.innerHTML = '<span class="loading"></span> Checking status...';
        button.disabled = true;
        
        try {
            const isAvailable = await window.checkStreamlitStatus();
            
            if (isAvailable) {
                button.innerHTML = '<i class="fas fa-external-link-alt me-2"></i>Launching...';
                
                // Open Streamlit app
                const streamlitUrl = 'https://oracle-samuel-app.streamlit.app/';
                window.open(streamlitUrl, '_blank');
                
                // Track launch event
                trackEvent('streamlit_app_launched', {
                    source: 'github_pages',
                    timestamp: new Date().toISOString()
                });
            } else {
                button.innerHTML = '<i class="fas fa-exclamation-triangle me-2"></i>App Unavailable';
                button.classList.add('btn-warning');
                
                // Show fallback message
                showNotification('Streamlit app is currently unavailable. Please try again later.', 'warning');
            }
        } catch (error) {
            button.innerHTML = '<i class="fas fa-exclamation-triangle me-2"></i>Error';
            button.classList.add('btn-danger');
            showNotification('Unable to connect to the application. Please check your internet connection.', 'error');
        }
        
        setTimeout(() => {
            button.innerHTML = originalText;
            button.disabled = false;
            button.classList.remove('btn-warning', 'btn-danger');
        }, 3000);
    };
}

// Performance optimizations
function setupPerformanceOptimizations() {
    // Lazy load images
    const images = document.querySelectorAll('img[data-src]');
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.classList.remove('lazy');
                imageObserver.unobserve(img);
            }
        });
    });
    
    images.forEach(img => imageObserver.observe(img));
    
    // Preload critical resources
    preloadCriticalResources();
}

// Preload critical resources
function preloadCriticalResources() {
    const criticalResources = [
        'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css',
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
    ];
    
    criticalResources.forEach(resource => {
        const link = document.createElement('link');
        link.rel = 'preload';
        link.href = resource;
        link.as = 'style';
        document.head.appendChild(link);
    });
}

// Notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = `
        top: 100px;
        right: 20px;
        z-index: 9999;
        min-width: 300px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    `;
    
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

// Analytics tracking
function trackEvent(eventName, properties = {}) {
    // Google Analytics 4 tracking
    if (typeof gtag !== 'undefined') {
        gtag('event', eventName, properties);
    }
    
    // Console log for development
    console.log('Event tracked:', eventName, properties);
}

// Utility functions
const utils = {
    // Debounce function
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    // Throttle function
    throttle: function(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },
    
    // Format number with commas
    formatNumber: function(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    },
    
    // Get device type
    getDeviceType: function() {
        const width = window.innerWidth;
        if (width < 768) return 'mobile';
        if (width < 1024) return 'tablet';
        return 'desktop';
    },
    
    // Check if element is in viewport
    isInViewport: function(element) {
        const rect = element.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
    }
};

// Export for use in other scripts
window.OracleSamuelUtils = utils;

// Error handling
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
    
    // Track error (if analytics is available)
    trackEvent('javascript_error', {
        message: e.message,
        filename: e.filename,
        lineno: e.lineno,
        colno: e.colno
    });
});

// Performance monitoring
window.addEventListener('load', function() {
    // Track page load performance
    const perfData = performance.getEntriesByType('navigation')[0];
    
    trackEvent('page_load_performance', {
        load_time: perfData.loadEventEnd - perfData.loadEventStart,
        dom_content_loaded: perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart,
        device_type: utils.getDeviceType()
    });
});

// Service Worker registration (for PWA capabilities)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registration successful');
            })
            .catch(function(err) {
                console.log('ServiceWorker registration failed');
            });
    });
}

// Keyboard navigation support
document.addEventListener('keydown', function(e) {
    // ESC key to close modals/alerts
    if (e.key === 'Escape') {
        const activeAlert = document.querySelector('.alert.show');
        if (activeAlert) {
            const closeBtn = activeAlert.querySelector('.btn-close');
            if (closeBtn) closeBtn.click();
        }
    }
    
    // Enter key on buttons
    if (e.key === 'Enter' && e.target.tagName === 'BUTTON') {
        e.target.click();
    }
});

// Print functionality
window.printPage = function() {
    window.print();
};

// Share functionality
window.sharePage = function() {
    if (navigator.share) {
        navigator.share({
            title: 'Oracle Samuel - Universal AI Platform',
            text: 'Check out this amazing AI platform for market analysis!',
            url: window.location.href
        });
    } else {
        // Fallback: copy URL to clipboard
        navigator.clipboard.writeText(window.location.href).then(() => {
            showNotification('URL copied to clipboard!', 'success');
        });
    }
};

console.log('🚀 Oracle Samuel GitHub Pages initialized successfully!');
