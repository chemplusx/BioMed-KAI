---
layout: default
title: Analysis
---

<div class="interactive-analysis">
{% raw %}
<!-- Navigation - matching your main site -->
    <nav class="nav-glass fixed w-full top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between items-center py-4">
                <div class="flex items-center space-x-3">
                    <div class="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                        <i class="fas fa-brain text-white text-lg"></i>
                    </div>
                    <div>
                        <h1 class="text-xl font-bold text-gray-900">BioMed-KAI</h1>
                        <p class="text-xs text-gray-600">Analysis</p>
                    </div>
                </div>
                
                <div class="hidden md:flex space-x-8">
                    <a href="index.html" class="text-gray-600 hover:text-blue-600 px-3 py-2 text-sm font-medium transition-colors">Home</a>
                    <a href="analysis.html" class="nav-active px-3 py-2 text-sm font-medium transition-colors">Analysis</a>
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
                    <a href="index.html" class="block px-3 py-2 text-gray-600 hover:text-blue-600">Home</a>
                    <a href="analysis.html" class="block px-3 py-2 text-blue-600 font-medium">Analysis</a>
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
    <section class="pt-24 pb-16 px-4 sm:px-6 lg:px-8">
        <div class="max-w-7xl mx-auto">
            <div class="text-center mb-16">
                <h1 class="text-5xl font-bold text-gray-900 mb-6">
                    Analysis <span class="text-blue-600">Results</span>
                </h1>
                <p class="text-xl text-gray-600 mb-4 max-w-3xl mx-auto">
                    Comprehensive statistical validation and performance analysis of the BioMed-KAI system
                </p>
                <p class="text-lg text-gray-500 max-w-4xl mx-auto leading-relaxed">
                    Complete ablation studies, comparative benchmarks, and interactive visualizations 
                    from our research paper with statistical significance testing
                </p>
            </div>

            <!-- Key Metrics -->
            <div class="grid grid-cols-2 md:grid-cols-4 gap-6 mb-16">
                <div class="stat-card text-center p-6 rounded-2xl">
                    <div class="icon-container w-12 h-12 mx-auto mb-4">
                        <i class="fas fa-bullseye text-xl"></i>
                    </div>
                    <div class="text-3xl font-bold text-gray-900 mb-2">85.7%</div>
                    <div class="text-sm text-gray-600">Diagnostic Accuracy</div>
                </div>
                
                <div class="stat-card text-center p-6 rounded-2xl">
                    <div class="icon-container w-12 h-12 mx-auto mb-4">
                        <i class="fas fa-tachometer-alt text-xl"></i>
                    </div>
                    <div class="text-3xl font-bold text-gray-900 mb-2">66.5%</div>
                    <div class="text-sm text-gray-600">Token Efficiency</div>
                </div>
                
                <div class="stat-card text-center p-6 rounded-2xl">
                    <div class="icon-container w-12 h-12 mx-auto mb-4">
                        <i class="fas fa-chart-line text-xl"></i>
                    </div>
                    <div class="text-3xl font-bold text-gray-900 mb-2">39.3%</div>
                    <div class="text-sm text-gray-600">CARE-RAG Improvement</div>
                </div>
                
                <div class="stat-card text-center p-6 rounded-2xl">
                    <div class="icon-container w-12 h-12 mx-auto mb-4">
                        <i class="fas fa-certificate text-xl"></i>
                    </div>
                    <div class="text-3xl font-bold text-gray-900 mb-2">p < 0.001</div>
                    <div class="text-sm text-gray-600">Statistical Significance</div>
                </div>
            </div>
        </div>
    </section>

    <!-- Interactive Dashboards -->
    <section class="py-20 px-4 sm:px-6 lg:px-8">
        <div class="max-w-7xl mx-auto">
            <div class="text-center mb-16">
                <h2 class="text-4xl font-bold text-gray-900 mb-4">Interactive Analysis Dashboards</h2>
                <p class="text-xl text-gray-600">Explore our comprehensive analysis results through interactive visualizations</p>
            </div>

            <div class="analysis-grid">
                <!-- Performance Dashboard -->
                <div class="card-light rounded-2xl overflow-hidden">
                    <div class="dashboard-header">
                        <div class="flex items-center">
                            <div class="w-12 h-12 bg-white bg-opacity-20 rounded-lg flex items-center justify-center mr-4">
                                <i class="fas fa-chart-bar text-white text-xl"></i>
                            </div>
                            <div>
                                <h3 class="text-xl font-bold">Performance Dashboard</h3>
                                <p class="text-blue-100">Agent performance and CARE-RAG analysis</p>
                            </div>
                        </div>
                    </div>
                    <div class="p-6">
                        <p class="text-gray-600 mb-6">
                            Interactive overview of all system performance metrics, agent comparisons, 
                            and CARE-RAG ablation results with statistical validation.
                        </p>
                        <div class="flex flex-wrap gap-2 mb-6">
                            <span class="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">Agent Analysis</span>
                            <span class="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm">Performance Metrics</span>
                            <span class="bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm">Statistical Tests</span>
                        </div>
                        <a href="analysis/interactive/interactive_dashboard.html" 
                           class="btn-primary px-6 py-3 rounded-xl font-semibold inline-flex items-center">
                            <i class="fas fa-external-link-alt mr-2"></i>
                            View Dashboard
                        </a>
                    </div>
                </div>

                <!-- Error Analysis -->
                <div class="card-light rounded-2xl overflow-hidden">
                    <div class="dashboard-header">
                        <div class="flex items-center">
                            <div class="w-12 h-12 bg-white bg-opacity-20 rounded-lg flex items-center justify-center mr-4">
                                <i class="fas fa-exclamation-triangle text-white text-xl"></i>
                            </div>
                            <div>
                                <h3 class="text-xl font-bold">Error Analysis</h3>
                                <p class="text-blue-100">Failure patterns and recovery strategies</p>
                            </div>
                        </div>
                    </div>
                    <div class="p-6">
                        <p class="text-gray-600 mb-6">
                            Comprehensive analysis of error patterns, recovery rates, and mitigation strategies 
                            with interactive bubble charts and severity scoring.
                        </p>
                        <div class="flex flex-wrap gap-2 mb-6">
                            <span class="bg-red-100 text-red-800 px-3 py-1 rounded-full text-sm">Error Patterns</span>
                            <span class="bg-orange-100 text-orange-800 px-3 py-1 rounded-full text-sm">Recovery Rates</span>
                            <span class="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm">Mitigation</span>
                        </div>
                        <a href="analysis/interactive/interactive_error_analysis.html" 
                           class="btn-primary px-6 py-3 rounded-xl font-semibold inline-flex items-center">
                            <i class="fas fa-external-link-alt mr-2"></i>
                            View Analysis
                        </a>
                    </div>
                </div>

                <!-- Temperature Analysis -->
                <div class="card-light rounded-2xl overflow-hidden">
                    <div class="dashboard-header">
                        <div class="flex items-center">
                            <div class="w-12 h-12 bg-white bg-opacity-20 rounded-lg flex items-center justify-center mr-4">
                                <i class="fas fa-thermometer-half text-white text-xl"></i>
                            </div>
                            <div>
                                <h3 class="text-xl font-bold">Temperature Analysis</h3>
                                <p class="text-blue-100">Sensitivity and optimization results</p>
                            </div>
                        </div>
                    </div>
                    <div class="p-6">
                        <p class="text-gray-600 mb-6">
                            3D visualization of temperature optimization and performance sensitivity analysis 
                            with multi-dimensional parameter exploration.
                        </p>
                        <div class="flex flex-wrap gap-2 mb-6">
                            <span class="bg-indigo-100 text-indigo-800 px-3 py-1 rounded-full text-sm">3D Analysis</span>
                            <span class="bg-teal-100 text-teal-800 px-3 py-1 rounded-full text-sm">Optimization</span>
                            <span class="bg-cyan-100 text-cyan-800 px-3 py-1 rounded-full text-sm">Sensitivity</span>
                        </div>
                        <a href="analysis/interactive/interactive_temperature_analysis.html" 
                           class="btn-primary px-6 py-3 rounded-xl font-semibold inline-flex items-center">
                            <i class="fas fa-external-link-alt mr-2"></i>
                            View Analysis
                        </a>
                    </div>
                </div>

                <!-- Static Visualizations -->
                <div class="card-light rounded-2xl overflow-hidden">
                    <div class="dashboard-header">
                        <div class="flex items-center">
                            <div class="w-12 h-12 bg-white bg-opacity-20 rounded-lg flex items-center justify-center mr-4">
                                <i class="fas fa-images text-white text-xl"></i>
                            </div>
                            <div>
                                <h3 class="text-xl font-bold">Static Visualizations</h3>
                                <p class="text-blue-100">Publication-ready figures and charts</p>
                            </div>
                        </div>
                    </div>
                    <div class="p-6">
                        <p class="text-gray-600 mb-6">
                            High-resolution static visualizations including architecture diagrams, 
                            performance comparisons, and statistical validation plots.
                        </p>
                        <div class="flex flex-wrap gap-2 mb-6">
                            <span class="bg-gray-100 text-gray-800 px-3 py-1 rounded-full text-sm">Architecture</span>
                            <span class="bg-pink-100 text-pink-800 px-3 py-1 rounded-full text-sm">Comparisons</span>
                            <span class="bg-emerald-100 text-emerald-800 px-3 py-1 rounded-full text-sm">Statistics</span>
                        </div>
                        <a href="analysis/static/" 
                           class="btn-primary px-6 py-3 rounded-xl font-semibold inline-flex items-center">
                            <i class="fas fa-external-link-alt mr-2"></i>
                            View Gallery
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Research Validation -->
    <section class="py-20 px-4 sm:px-6 lg:px-8 bg-gray-50">
        <div class="max-w-7xl mx-auto">
            <div class="text-center mb-16">
                <h2 class="text-4xl font-bold text-gray-900 mb-4">Research Validation</h2>
                <p class="text-xl text-gray-600">Rigorous statistical methodology and reproducible results</p>
            </div>

            <div class="grid md:grid-cols-2 gap-8">
                <div class="card-light p-8 rounded-2xl">
                    <h3 class="text-2xl font-bold text-gray-900 mb-4">Statistical Framework</h3>
                    <ul class="space-y-3 text-gray-600">
                        <li class="flex items-start">
                            <i class="fas fa-check-circle text-blue-600 mr-3 mt-0.5"></i>
                            <span>Bootstrap validation with 10,000 iterations</span>
                        </li>
                        <li class="flex items-start">
                            <i class="fas fa-check-circle text-blue-600 mr-3 mt-0.5"></i>
                            <span>Multiple comparison correction (Benjamini-Hochberg)</span>
                        </li>
                        <li class="flex items-start">
                            <i class="fas fa-check-circle text-blue-600 mr-3 mt-0.5"></i>
                            <span>Effect size calculation (Cohen's d)</span>
                        </li>
                        <li class="flex items-start">
                            <i class="fas fa-check-circle text-blue-600 mr-3 mt-0.5"></i>
                            <span>Power analysis and sample size validation</span>
                        </li>
                    </ul>
                </div>

                <div class="card-light p-8 rounded-2xl">
                    <h3 class="text-2xl font-bold text-gray-900 mb-4">Reproducibility</h3>
                    <p class="text-gray-600 mb-6">
                        All analysis scripts and data are openly available. Our comprehensive 
                        analysis suite generates these interactive visualizations automatically 
                        from the experimental results.
                    </p>
                    <div class="flex flex-col sm:flex-row gap-4">
                        <a href="https://github.com/chemplusx/BioMed-KAI" target="_blank" 
                           class="bg-gray-900 text-white px-6 py-3 rounded-xl font-semibold hover:bg-gray-800 transition-colors inline-flex items-center justify-center">
                            <i class="fab fa-github mr-2"></i>
                            View Code
                        </a>
                        <a href="https://github.com/chemplusx/BioMed-KAI/releases" target="_blank" 
                           class="border-2 border-gray-300 text-gray-700 px-6 py-3 rounded-xl font-semibold hover:bg-gray-50 transition-colors inline-flex items-center justify-center">
                            <i class="fas fa-download mr-2"></i>
                            Download Data
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Footer matching main site -->
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
                            <p class="text-sm text-gray-600">Analysis Results</p>
                        </div>
                    </div>
                    <p class="text-gray-600 leading-relaxed">
                        Comprehensive analysis results for the BioMed-KAI research project, 
                        demonstrating significant improvements in biomedical AI performance.
                    </p>
                </div>
                
                <div>
                    <h4 class="font-semibold text-gray-900 mb-4">Analysis</h4>
                    <ul class="space-y-2">
                        <li><a href="analysis/interactive/interactive_dashboard.html" class="text-gray-600 hover:text-blue-600 transition-colors">Performance Dashboard</a></li>
                        <li><a href="analysis/interactive/interactive_error_analysis.html" class="text-gray-600 hover:text-blue-600 transition-colors">Error Analysis</a></li>
                        <li><a href="analysis/interactive/interactive_temperature_analysis.html" class="text-gray-600 hover:text-blue-600 transition-colors">Temperature Analysis</a></li>
                    </ul>
                </div>
                
                <div>
                    <h4 class="font-semibold text-gray-900 mb-4">Resources</h4>
                    <ul class="space-y-2">
                        <li><a href="index.html" class="text-gray-600 hover:text-blue-600 transition-colors">Platform</a></li>
                        <li><a href="https://github.com/chemplusx/BioMed-KAI" target="_blank" class="text-gray-600 hover:text-blue-600 transition-colors">GitHub</a></li>
                        <li><a href="about.html" class="text-gray-600 hover:text-blue-600 transition-colors">About</a></li>
                    </ul>
                </div>
            </div>
            
            <div class="border-t border-gray-200 pt-8 text-center">
                <p class="text-gray-600">
                    &copy; 2025 BioMed-KAI Research Project. Advancing Precision Medicine through Agentic AI.
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

        // Smooth scrolling
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            });
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
