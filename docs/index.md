---
layout: default
title: Index
---

<div class="interactive-analysis">
{% raw %}
<!-- Navigation -->
    <nav class="nav-glass fixed w-full top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between items-center py-4">
                <div class="flex items-center space-x-3">
                    <div class="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                        <i class="fas fa-brain text-white text-lg"></i>
                    </div>
                    <div>
                        <h1 class="text-xl font-bold text-gray-900">BioMed-KAI</h1>
                        <p class="text-xs text-gray-600">Showcase</p>
                    </div>
                </div>
                
                <div class="hidden md:flex space-x-8">
                    <a href="index.html" class="nav-active px-3 py-2 text-sm font-medium transition-colors">Home</a>
                    <a href="analysis.html" class="text-gray-600 hover:text-blue-600 px-3 py-2 text-sm font-medium transition-colors">Analysis</a>
                    <a href="sources.html" class="text-gray-600 hover:text-blue-600 px-3 py-2 text-sm font-medium transition-colors">Sources</a>
                    <a href="about.html" class="text-gray-600 hover:text-blue-600 px-3 py-2 text-sm font-medium transition-colors">About</a>
                    <a href="https://github.com/chemplusx/BioMed-KAI" target="_blank" class="text-gray-600 hover:text-blue-600 px-3 py-2 text-sm font-medium transition-colors">
                        <i class="fab fa-github mr-1"></i>GitHub
                    </a>
                </div>

                <!-- Mobile menu button -->
                <div class="md:hidden">
                    <button id="mobile-menu-button" class="text-gray-600 hover:text-blue-600">
                        <i class="fas fa-bars text-lg"></i>
                    </button>
                </div>
            </div>

            <!-- Mobile menu -->
            <div id="mobile-menu" class="hidden md:hidden pb-4">
                <div class="space-y-2">
                    <a href="index.html" class="block px-3 py-2 text-blue-600 font-medium">Home</a>
                    <a href="analysis.html" class="block px-3 py-2 text-blue-600 hover:text-blue-600">Analysis</a>
                    <a href="sources.html" class="block px-3 py-2 text-gray-600 hover:text-blue-600">Sources</a>
                    <a href="about.html" class="block px-3 py-2 text-gray-600 hover:text-blue-600">About</a>
                    <a href="https://github.com/chemplusx/BioMed-KAI" target="_blank" class="block px-3 py-2 text-gray-600 hover:text-blue-600">
                        <i class="fab fa-github mr-1"></i>GitHub
                    </a>
                </div>
            </div>
        </div>
    </nav>

    <!-- Hero Section -->
    <section class="hero-gradient pt-24 pb-16 px-4 sm:px-6 lg:px-8">
        <!-- Floating particles -->
        <div class="particle w-4 h-4 top-20 left-1/4"></div>
        <div class="particle w-6 h-6 top-40 right-1/4"></div>
        <div class="particle w-3 h-3 top-60 left-1/3"></div>
        <div class="particle w-5 h-5 top-32 right-1/3"></div>
        
        <div class="max-w-7xl mx-auto relative z-10">
            <div class="text-center">
                <div class="fade-in mb-8">
                    <div class="inline-flex items-center bg-white rounded-full px-6 py-3 shadow-lg border border-gray-200 mb-8">
                        <span class="w-2 h-2 bg-green-500 rounded-full mr-3 animate-pulse"></span>
                        <span class="text-sm font-medium text-gray-700">Live Demo Available</span>
                        <i class="fas fa-external-link-alt ml-2 text-blue-600 text-xs"></i>
                    </div>
                </div>

                <h1 class="hero-title text-5xl md:text-7xl font-bold text-gray-900 mb-6 fade-in">
                    BioMed-<span class="text-blue-600">KAI</span>
                </h1>
                
                <p class="hero-subtitle text-xl md:text-2xl text-gray-600 mb-4 fade-in-delay max-w-3xl mx-auto">
                    Advanced Agentic AI Platform for Precision Medicine
                </p>
                
                <p class="text-lg text-gray-500 mb-12 fade-in-delay-2 max-w-4xl mx-auto leading-relaxed">
                    Revolutionizing healthcare through <strong>intelligent AI agents</strong>, 
                    <strong>RAG-enhanced inference</strong>, and <strong>dynamic knowledge graphs</strong> 
                    with 200M+ medical relationships
                </p>

                <div class="flex flex-col sm:flex-row gap-4 justify-center items-center fade-in-delay-2">
                    <a href="http://pitools.niper.ac.in/biomedkai/home" target="_blank" 
                       class="btn-primary px-8 py-4 rounded-xl font-semibold inline-flex items-center">
                        <i class="fas fa-rocket mr-3"></i>
                        Launch Platform
                    </a>
                    <a href="http://pitools.niper.ac.in/biomedkai/midas" target="_blank" 
                       class="btn-secondary px-8 py-4 rounded-xl font-semibold inline-flex items-center">
                        <i class="fas fa-comments mr-3"></i>
                        Try AI Chat
                    </a>
                </div>
            </div>
        </div>
    </section>

    <!-- Stats Section -->
    <section class="py-16 px-4 sm:px-6 lg:px-8">
        <div class="max-w-7xl mx-auto">
            <div class="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div class="stat-card text-center p-6 rounded-2xl">
                    <div class="icon-container w-12 h-12 mx-auto mb-4">
                        <i class="fas fa-database text-xl"></i>
                    </div>
                    <div class="text-3xl font-bold text-gray-900 mb-2">21</div>
                    <div class="text-sm text-gray-600">Data Sources</div>
                </div>
                
                <div class="stat-card text-center p-6 rounded-2xl">
                    <div class="icon-container w-12 h-12 mx-auto mb-4">
                        <i class="fas fa-project-diagram text-xl"></i>
                    </div>
                    <div class="text-3xl font-bold text-gray-900 mb-2">200M+</div>
                    <div class="text-sm text-gray-600">Relationships</div>
                </div>
                
                <div class="stat-card text-center p-6 rounded-2xl">
                    <div class="icon-container w-12 h-12 mx-auto mb-4">
                        <i class="fas fa-robot text-xl"></i>
                    </div>
                    <div class="text-3xl font-bold text-gray-900 mb-2">Multi</div>
                    <div class="text-sm text-gray-600">AI Agents</div>
                </div>
                
                <div class="stat-card text-center p-6 rounded-2xl">
                    <div class="icon-container w-12 h-12 mx-auto mb-4">
                        <i class="fas fa-brain text-xl"></i>
                    </div>
                    <div class="text-3xl font-bold text-gray-900 mb-2">15</div>
                    <div class="text-sm text-gray-600">Ontologies</div>
                </div>
            </div>
        </div>
    </section>

    <!-- Core Features -->
    <section class="py-20 px-4 sm:px-6 lg:px-8">
        <div class="max-w-7xl mx-auto">
            <div class="text-center mb-16">
                <h2 class="text-4xl font-bold text-gray-900 mb-4">Core Capabilities</h2>
                <p class="text-xl text-gray-600 max-w-3xl mx-auto">
                    Discover how BioMed-KAI is transforming precision medicine through advanced AI
                </p>
            </div>

            <div class="grid md:grid-cols-3 gap-8">
                <!-- Agentic AI -->
                <div class="card-light p-8 rounded-2xl">
                    <div class="icon-container w-16 h-16 mb-6">
                        <i class="fas fa-robot text-2xl"></i>
                    </div>
                    <h3 class="text-2xl font-bold text-gray-900 mb-4">Agentic AI System</h3>
                    <p class="text-gray-600 mb-6 leading-relaxed">
                        Autonomous AI agents that collaborate, reason, and learn from interactions. 
                        Our multi-agent architecture provides comprehensive biomedical analysis 
                        with human-level expertise across medical domains.
                    </p>
                    <div class="feature-highlight p-4 rounded-xl">
                        <ul class="text-sm text-gray-700 space-y-2">
                            <li><i class="fas fa-check text-blue-600 mr-2"></i>Multi-agent collaboration</li>
                            <li><i class="fas fa-check text-blue-600 mr-2"></i>Self-reflective learning</li>
                            <li><i class="fas fa-check text-blue-600 mr-2"></i>Autonomous planning</li>
                        </ul>
                    </div>
                </div>

                <!-- RAG Intelligence -->
                <div class="card-light p-8 rounded-2xl">
                    <div class="icon-container w-16 h-16 mb-6">
                        <i class="fas fa-search-plus text-2xl"></i>
                    </div>
                    <h3 class="text-2xl font-bold text-gray-900 mb-4">RAG-Enhanced Intelligence</h3>
                    <p class="text-gray-600 mb-6 leading-relaxed">
                        Retrieval-Augmented Generation providing context-aware responses grounded 
                        in the latest medical literature. Dynamic knowledge retrieval ensures 
                        accurate, up-to-date medical information.
                    </p>
                    <div class="feature-highlight p-4 rounded-xl">
                        <ul class="text-sm text-gray-700 space-y-2">
                            <li><i class="fas fa-check text-blue-600 mr-2"></i>Real-time literature access</li>
                            <li><i class="fas fa-check text-blue-600 mr-2"></i>Context-aware responses</li>
                            <li><i class="fas fa-check text-blue-600 mr-2"></i>Semantic search capabilities</li>
                        </ul>
                    </div>
                </div>

                <!-- Knowledge Graph -->
                <div class="card-light p-8 rounded-2xl">
                    <div class="icon-container w-16 h-16 mb-6">
                        <i class="fas fa-project-diagram text-2xl"></i>
                    </div>
                    <h3 class="text-2xl font-bold text-gray-900 mb-4">Dynamic Knowledge Graph</h3>
                    <p class="text-gray-600 mb-6 leading-relaxed">
                        Neo4j-powered knowledge graph with 200M+ relationships across diseases, 
                        drugs, genes, and clinical outcomes. Integrates 21 databases and 15 ontologies 
                        for comprehensive biomedical knowledge.
                    </p>
                    <div class="feature-highlight p-4 rounded-xl">
                        <ul class="text-sm text-gray-700 space-y-2">
                            <li><i class="fas fa-check text-blue-600 mr-2"></i>Real-time updates</li>
                            <li><i class="fas fa-check text-blue-600 mr-2"></i>Comprehensive integration</li>
                            <li><i class="fas fa-check text-blue-600 mr-2"></i>Semantic relationships</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Technology Stack -->
    <section class="py-20 px-4 sm:px-6 lg:px-8 bg-gray-50">
        <div class="max-w-7xl mx-auto">
            <div class="text-center mb-16">
                <h2 class="text-4xl font-bold text-gray-900 mb-4">Built with Modern Technology</h2>
                <p class="text-xl text-gray-600">Cutting-edge tools for performance and reliability</p>
            </div>

            <div class="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div class="card-light p-6 rounded-xl text-center">
                    <i class="fab fa-python text-4xl text-blue-600 mb-3"></i>
                    <h4 class="font-semibold text-gray-900">Python</h4>
                    <p class="text-sm text-gray-600">Agentic Backend</p>
                </div>
                
                <div class="card-light p-6 rounded-xl text-center">
                    <i class="fab fa-golang text-4xl text-blue-600 mb-3"></i>
                    <h4 class="font-semibold text-gray-900">Go</h4>
                    <p class="text-sm text-gray-600">Web Server</p>
                </div>
                
                <div class="card-light p-6 rounded-xl text-center">
                    <i class="fas fa-database text-4xl text-blue-600 mb-3"></i>
                    <h4 class="font-semibold text-gray-900">Neo4j</h4>
                    <p class="text-sm text-gray-600">Knowledge Graph</p>
                </div>
                
                <div class="card-light p-6 rounded-xl text-center">
                    <i class="fab fa-docker text-4xl text-blue-600 mb-3"></i>
                    <h4 class="font-semibold text-gray-900">Docker</h4>
                    <p class="text-sm text-gray-600">Deployment</p>
                </div>
            </div>
        </div>
    </section>

    <!-- Call to Action -->
    <section class="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-blue-600 to-purple-700">
        <div class="max-w-4xl mx-auto text-center">
            <h2 class="text-4xl md:text-5xl font-bold text-white mb-6">
                Ready to Explore BioMed-KAI?
            </h2>
            <p class="text-xl text-blue-100 mb-8 leading-relaxed">
                Experience the future of precision medicine through advanced agentic AI and intelligent knowledge graphs
            </p>
            <div class="flex flex-col sm:flex-row gap-4 justify-center">
                <a href="http://pitools.niper.ac.in/biomedkai/home" target="_blank" 
                   class="bg-white text-blue-600 px-8 py-4 rounded-xl font-semibold hover:bg-gray-100 transition-colors inline-flex items-center justify-center">
                    <i class="fas fa-external-link-alt mr-3"></i>
                    Visit Live Platform
                </a>
                <a href="sources.html" 
                   class="border-2 border-white text-white px-8 py-4 rounded-xl font-semibold hover:bg-white hover:text-blue-600 transition-colors inline-flex items-center justify-center">
                    <i class="fas fa-database mr-3"></i>
                    Explore Data Sources
                </a>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer class="bg-white py-12 px-4 sm:px-6 lg:px-8">
        <div class="max-w-7xl mx-auto">
            <div class="grid md:grid-cols-4 gap-8 mb-8">
                <div class="md:col-span-2">
                    <div class="flex items-center space-x-3 mb-4">
                        <div class="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                            <i class="fas fa-brain text-white"></i>
                        </div>
                        <div>
                            <h3 class="text-lg font-bold text-gray-900">BioMed-KAI</h3>
                            <p class="text-sm text-gray-600">Showcase</p>
                        </div>
                    </div>
                    <p class="text-gray-600 leading-relaxed mb-4">
                        An advanced agentic AI platform revolutionizing precision medicine through 
                        intelligent knowledge graphs and RAG-enhanced inference.
                    </p>
                    <div class="flex space-x-4">
                        <a href="https://github.com/chemplusx/BioMed-KAI" target="_blank" 
                           class="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center hover:bg-gray-200 transition-colors">
                            <i class="fab fa-github text-gray-700"></i>
                        </a>
                    </div>
                </div>
                
                <div>
                    <h4 class="font-semibold text-gray-900 mb-4">Platform</h4>
                    <ul class="space-y-2">
                        <li><a href="http://pitools.niper.ac.in/biomedkai/home" target="_blank" class="text-gray-600 hover:text-blue-600 transition-colors">Live Demo</a></li>
                        <li><a href="http://pitools.niper.ac.in/biomedkai/midas" target="_blank" class="text-gray-600 hover:text-blue-600 transition-colors">AI Chat</a></li>
                        <li><a href="sources.html" class="text-gray-600 hover:text-blue-600 transition-colors">Data Sources</a></li>
                    </ul>
                </div>
                
                <div>
                    <h4 class="font-semibold text-gray-900 mb-4">Project</h4>
                    <ul class="space-y-2">
                        <li><a href="about.html" class="text-gray-600 hover:text-blue-600 transition-colors">About</a></li>
                        <li><a href="https://github.com/chemplusx/BioMed-KAI" target="_blank" class="text-gray-600 hover:text-blue-600 transition-colors">GitHub</a></li>
                        <li><a href="https://github.com/chemplusx/BioMed-KAI/blob/master/README.md" target="_blank" class="text-gray-600 hover:text-blue-600 transition-colors">Documentation</a></li>
                    </ul>
                </div>
            </div>
            
            <div class="border-t border-gray-200 pt-8 text-center">
                <p class="text-gray-600">
                    &copy; 2024 BioMed-KAI Research Project. Advancing Precision Medicine through Agentic AI.
                </p>
            </div>
        </div>
    </footer>

    <script>
        // Mobile menu toggle
        document.getElementById('mobile-menu-button').addEventListener('click', function() {
            const menu = document.getElementById('mobile-menu');
            menu.classList.toggle('hidden');
        });

        // Smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            });
        });

        // Intersection Observer for animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver(function(entries) {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, observerOptions);

        // Observe fade-in elements
        document.querySelectorAll('.fade-in, .fade-in-delay, .fade-in-delay-2').forEach(el => {
            observer.observe(el);
        });
    </script>
{% endraw %}
</div>

<style>
.interactive-analysis {
    width: 100%;
    margin: 20px 0;
}

.interactive-analysis .plotly-graph-div {
    width: 100% !important;
    height: auto !important;
}
</style>
