---
layout: default
title: About
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
                    <a href="index.html" class="text-gray-600 hover:text-blue-600 px-3 py-2 text-sm font-medium transition-colors">Home</a>
                    <a href="analysis.html" class="text-gray-600 hover:text-blue-600 px-3 py-2 text-sm font-medium transition-colors">Analysis</a>
                    <a href="sources.html" class="text-gray-600 hover:text-blue-600 px-3 py-2 text-sm font-medium transition-colors">Sources</a>
                    <a href="about.html" class="nav-active px-3 py-2 text-sm font-medium transition-colors">About</a>
                    <a href="https://github.com/chemplusx/BioMed-KAI" target="_blank" class="text-gray-600 hover:text-blue-600 px-3 py-2 text-sm font-medium transition-colors">
                        <i class="fab fa-github mr-1"></i>GitHub
                    </a>
                </div>

                <div class="md:hidden">
                    <button id="mobile-menu-button" class="text-gray-600 hover:text-blue-600">
                        <i class="fas fa-bars text-lg"></i>
                    </button>
                </div>
            </div>

            <div id="mobile-menu" class="hidden md:hidden pb-4">
                <div class="space-y-2">
                    <a href="index.html" class="block px-3 py-2 text-gray-600 hover:text-blue-600">Home</a>
                    <a href="analysis.html" class="block px-3 py-2 text-blue-600 hover:text-blue-600">Analysis</a>
                    <a href="sources.html" class="block px-3 py-2 text-gray-600 hover:text-blue-600">Sources</a>
                    <a href="about.html" class="block px-3 py-2 text-blue-600 font-medium">About</a>
                    <a href="https://github.com/chemplusx/BioMed-KAI" target="_blank" class="block px-3 py-2 text-gray-600 hover:text-blue-600">
                        <i class="fab fa-github mr-1"></i>GitHub
                    </a>
                </div>
            </div>
        </div>
    </nav>

    <!-- Header -->
    <section class="pt-24 pb-16 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-blue-50 to-purple-50">
        <div class="max-w-4xl mx-auto text-center">
            <h1 class="text-4xl md:text-5xl font-bold text-gray-900 mb-6 fade-in">
                About BioMed-KAI
            </h1>
            <p class="text-xl text-gray-600 mb-8 leading-relaxed fade-in-delay">
                Pioneering the future of precision medicine through advanced agentic AI, 
                intelligent knowledge graphs, and collaborative research
            </p>
            
            <div class="research-highlight rounded-2xl p-6 fade-in-delay-2">
                <div class="flex items-center justify-center mb-4">
                    <i class="fas fa-university text-3xl text-orange-600 mr-3"></i>
                    <div class="text-left">
                        <h3 class="text-lg font-bold text-gray-900">Research Project</h3>
                        <p class="text-sm text-gray-600">Advancing AI in Healthcare</p>
                    </div>
                </div>
                <p class="text-gray-700">
                    This project represents cutting-edge research in applying agentic AI systems 
                    to precision medicine, conducted by a multidisciplinary team of researchers.
                </p>
            </div>
        </div>
    </section>

    <!-- Mission & Vision -->
    <section class="py-16 px-4 sm:px-6 lg:px-8">
        <div class="max-w-7xl mx-auto">
            <div class="grid md:grid-cols-2 gap-12">
                <div class="card-light rounded-2xl p-8">
                    <div class="icon-bg w-16 h-16 rounded-2xl flex items-center justify-center mb-6">
                        <i class="fas fa-bullseye text-2xl"></i>
                    </div>
                    <h2 class="text-2xl font-bold text-gray-900 mb-4">Our Mission</h2>
                    <p class="text-gray-600 leading-relaxed mb-6">
                        To revolutionize precision medicine by developing advanced agentic AI systems 
                        that can autonomously analyze, reason, and collaborate to provide intelligent 
                        healthcare decision support through comprehensive biomedical knowledge integration.
                    </p>
                    <div class="feature-highlight rounded-xl p-4">
                        <ul class="text-sm text-gray-700 space-y-2">
                            <li><i class="fas fa-check text-blue-600 mr-2"></i>Advance AI in healthcare</li>
                            <li><i class="fas fa-check text-blue-600 mr-2"></i>Enable precision medicine</li>
                            <li><i class="fas fa-check text-blue-600 mr-2"></i>Support clinical decisions</li>
                        </ul>
                    </div>
                </div>

                <div class="card-light rounded-2xl p-8">
                    <div class="icon-bg w-16 h-16 rounded-2xl flex items-center justify-center mb-6">
                        <i class="fas fa-eye text-2xl"></i>
                    </div>
                    <h2 class="text-2xl font-bold text-gray-900 mb-4">Our Vision</h2>
                    <p class="text-gray-600 leading-relaxed mb-6">
                        To create a future where AI agents work seamlessly alongside healthcare 
                        professionals, providing real-time, evidence-based insights that improve 
                        patient outcomes and accelerate medical research discovery.
                    </p>
                    <div class="feature-highlight rounded-xl p-4">
                        <ul class="text-sm text-gray-700 space-y-2">
                            <li><i class="fas fa-check text-blue-600 mr-2"></i>Human-AI collaboration</li>
                            <li><i class="fas fa-check text-blue-600 mr-2"></i>Improved patient outcomes</li>
                            <li><i class="fas fa-check text-blue-600 mr-2"></i>Accelerated research</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Research Team -->
    <section class="py-16 px-4 sm:px-6 lg:px-8 bg-gray-50">
        <div class="max-w-7xl mx-auto">
            <div class="text-center mb-16">
                <h2 class="text-3xl font-bold text-gray-900 mb-4">Research Team</h2>
                <p class="text-xl text-gray-600 max-w-3xl mx-auto">
                    Our multidisciplinary team combines expertise in AI, medicine, and data science 
                    to push the boundaries of what's possible in healthcare technology
                </p>
            </div>

            <div class="grid md:grid-cols-3 gap-8">
                <div class="team-card rounded-2xl p-8 text-center">
                    <div class="w-24 h-24 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-6">
                        <i class="fas fa-user-graduate text-white text-2xl"></i>
                    </div>
                    <h3 class="text-xl font-bold text-gray-900 mb-2">Madhavi Kumari</h3>
                    <p class="text-blue-600 font-medium mb-4">Lead Researcher</p>
                    <p class="text-gray-600 text-sm leading-relaxed">
                        Leading the research initiative in applying advanced AI methodologies 
                        to precision medicine and biomedical knowledge integration.
                    </p>
                </div>

                <div class="team-card rounded-2xl p-8 text-center">
                    <div class="w-24 h-24 bg-gradient-to-br from-green-500 to-blue-600 rounded-full flex items-center justify-center mx-auto mb-6">
                        <i class="fas fa-code text-white text-2xl"></i>
                    </div>
                    <h3 class="text-xl font-bold text-gray-900 mb-2">Rohit Chauhan</h3>
                    <p class="text-green-600 font-medium mb-4">Lead Data Scientist & Software Architect</p>
                    <p class="text-gray-600 text-sm leading-relaxed">
                        Architecting the agentic AI system, RAG infrastructure, and 
                        knowledge graph implementation for scalable biomedical applications.
                    </p>
                </div>

                <div class="team-card rounded-2xl p-8 text-center">
                    <div class="w-24 h-24 bg-gradient-to-br from-purple-500 to-pink-600 rounded-full flex items-center justify-center mx-auto mb-6">
                        <i class="fas fa-chalkboard-teacher text-white text-2xl"></i>
                    </div>
                    <h3 class="text-xl font-bold text-gray-900 mb-2">Prof. Prabha Garg</h3>
                    <p class="text-purple-600 font-medium mb-4">Project Guide</p>
                    <p class="text-gray-600 text-sm leading-relaxed">
                        Providing strategic guidance and academic oversight to ensure 
                        research excellence and alignment with healthcare innovation goals.
                    </p>
                </div>
            </div>
        </div>
    </section>

    <!-- Project Timeline -->
    <section class="py-16 px-4 sm:px-6 lg:px-8">
        <div class="max-w-4xl mx-auto">
            <div class="text-center mb-16">
                <h2 class="text-3xl font-bold text-gray-900 mb-4">Project Evolution</h2>
                <p class="text-xl text-gray-600">
                    Our journey in developing advanced agentic AI for precision medicine
                </p>
            </div>

            <div class="space-y-8">
                <div class="timeline-item">
                    <div class="card-light rounded-xl p-6">
                        <h3 class="text-xl font-bold text-gray-900 mb-2">Research Initiation</h3>
                        <p class="text-sm text-blue-600 font-medium mb-3">Phase 1: Foundation</p>
                        <p class="text-gray-600">
                            Established the research framework for applying agentic AI to precision medicine. 
                            Conducted comprehensive literature review and identified key challenges in 
                            biomedical knowledge integration.
                        </p>
                    </div>
                </div>

                <div class="timeline-item">
                    <div class="card-light rounded-xl p-6">
                        <h3 class="text-xl font-bold text-gray-900 mb-2">Knowledge Graph Development</h3>
                        <p class="text-sm text-blue-600 font-medium mb-3">Phase 2: Infrastructure</p>
                        <p class="text-gray-600">
                            Built comprehensive knowledge graph integrating 21 databases and 15 ontologies. 
                            Developed Neo4j-based architecture with 200M+ medical relationships for 
                            semantic biomedical knowledge representation.
                        </p>
                    </div>
                </div>

                <div class="timeline-item">
                    <div class="card-light rounded-xl p-6">
                        <h3 class="text-xl font-bold text-gray-900 mb-2">Agentic AI Implementation</h3>
                        <p class="text-sm text-blue-600 font-medium mb-3">Phase 3: Intelligence</p>
                        <p class="text-gray-600">
                            Implemented multi-agent system with RAG-enhanced capabilities. 
                            Developed autonomous AI agents capable of collaborative reasoning, 
                            self-reflection, and continuous learning in biomedical domains.
                        </p>
                    </div>
                </div>

                <div class="timeline-item">
                    <div class="card-light rounded-xl p-6">
                        <h3 class="text-xl font-bold text-gray-900 mb-2">Platform Integration</h3>
                        <p class="text-sm text-blue-600 font-medium mb-3">Phase 4: Deployment</p>
                        <p class="text-gray-600">
                            Deployed production-ready platform with real-time inference capabilities. 
                            Integrated Docker containerization, scalable architecture, and 
                            user-friendly interfaces for healthcare professionals.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Technical Innovation -->
    <section class="py-16 px-4 sm:px-6 lg:px-8 bg-gray-50">
        <div class="max-w-7xl mx-auto">
            <div class="text-center mb-16">
                <h2 class="text-3xl font-bold text-gray-900 mb-4">Technical Innovation</h2>
                <p class="text-xl text-gray-600">
                    Breakthrough technologies advancing the field of AI in healthcare
                </p>
            </div>

            <div class="grid md:grid-cols-2 gap-8">
                <div class="card-light rounded-2xl p-8">
                    <div class="icon-bg w-16 h-16 rounded-2xl flex items-center justify-center mb-6">
                        <i class="fas fa-robot text-2xl"></i>
                    </div>
                    <h3 class="text-2xl font-bold text-gray-900 mb-4">Agentic AI Architecture</h3>
                    <p class="text-gray-600 mb-4">
                        Novel implementation of autonomous AI agents that can plan, collaborate, 
                        and learn from medical data interactions.
                    </p>
                    <ul class="text-sm text-gray-600 space-y-2">
                        <li><i class="fas fa-cog text-blue-600 mr-2"></i>Multi-agent coordination</li>
                        <li><i class="fas fa-cog text-blue-600 mr-2"></i>Self-reflective learning</li>
                        <li><i class="fas fa-cog text-blue-600 mr-2"></i>Autonomous planning</li>
                    </ul>
                </div>

                <div class="card-light rounded-2xl p-8">
                    <div class="icon-bg w-16 h-16 rounded-2xl flex items-center justify-center mb-6">
                        <i class="fas fa-search-plus text-2xl"></i>
                    </div>
                    <h3 class="text-2xl font-bold text-gray-900 mb-4">RAG-Enhanced Intelligence</h3>
                    <p class="text-gray-600 mb-4">
                        Advanced retrieval-augmented generation system providing context-aware 
                        responses grounded in comprehensive medical literature.
                    </p>
                    <ul class="text-sm text-gray-600 space-y-2">
                        <li><i class="fas fa-cog text-blue-600 mr-2"></i>Dynamic knowledge retrieval</li>
                        <li><i class="fas fa-cog text-blue-600 mr-2"></i>Context-aware generation</li>
                        <li><i class="fas fa-cog text-blue-600 mr-2"></i>Semantic search capabilities</li>
                    </ul>
                </div>

                <div class="card-light rounded-2xl p-8">
                    <div class="icon-bg w-16 h-16 rounded-2xl flex items-center justify-center mb-6">
                        <i class="fas fa-project-diagram text-2xl"></i>
                    </div>
                    <h3 class="text-2xl font-bold text-gray-900 mb-4">Knowledge Graph Innovation</h3>
                    <p class="text-gray-600 mb-4">
                        Comprehensive biomedical knowledge graph with real-time updates 
                        and semantic relationship modeling.
                    </p>
                    <ul class="text-sm text-gray-600 space-y-2">
                        <li><i class="fas fa-cog text-blue-600 mr-2"></i>200M+ medical relationships</li>
                        <li><i class="fas fa-cog text-blue-600 mr-2"></i>Real-time data integration</li>
                        <li><i class="fas fa-cog text-blue-600 mr-2"></i>Semantic reasoning</li>
                    </ul>
                </div>

                <div class="card-light rounded-2xl p-8">
                    <div class="icon-bg w-16 h-16 rounded-2xl flex items-center justify-center mb-6">
                        <i class="fas fa-stethoscope text-2xl"></i>
                    </div>
                    <h3 class="text-2xl font-bold text-gray-900 mb-4">Clinical Integration</h3>
                    <p class="text-gray-600 mb-4">
                        Seamless integration with healthcare workflows providing 
                        intelligent decision support for precision medicine.
                    </p>
                    <ul class="text-sm text-gray-600 space-y-2">
                        <li><i class="fas fa-cog text-blue-600 mr-2"></i>Clinical decision support</li>
                        <li><i class="fas fa-cog text-blue-600 mr-2"></i>Workflow integration</li>
                        <li><i class="fas fa-cog text-blue-600 mr-2"></i>Evidence-based insights</li>
                    </ul>
                </div>
            </div>
        </div>
    </section>

    <!-- Impact & Future -->
    <section class="py-16 px-4 sm:px-6 lg:px-8">
        <div class="max-w-4xl mx-auto text-center">
            <h2 class="text-3xl font-bold text-gray-900 mb-8">Research Impact & Future Directions</h2>
            
            <div class="card-light rounded-2xl p-8 mb-8">
                <h3 class="text-xl font-bold text-gray-900 mb-4">Current Contributions</h3>
                <div class="grid md:grid-cols-3 gap-6 text-center">
                    <div>
                        <div class="text-3xl font-bold text-blue-600 mb-2">Novel</div>
                        <div class="text-sm text-gray-600">Agentic AI Architecture for Healthcare</div>
                    </div>
                    <div>
                        <div class="text-3xl font-bold text-green-600 mb-2">Comprehensive</div>
                        <div class="text-sm text-gray-600">Biomedical Knowledge Integration</div>
                    </div>
                    <div>
                        <div class="text-3xl font-bold text-purple-600 mb-2">Scalable</div>
                        <div class="text-sm text-gray-600">Precision Medicine Platform</div>
                    </div>
                </div>
            </div>

            <div class="feature-highlight rounded-2xl p-8">
                <h3 class="text-xl font-bold text-gray-900 mb-4">Future Research Directions</h3>
                <div class="grid md:grid-cols-2 gap-6 text-left">
                    <div>
                        <h4 class="font-semibold text-gray-900 mb-2">Enhanced AI Capabilities</h4>
                        <ul class="text-sm text-gray-600 space-y-1">
                            <li>• Advanced multi-modal reasoning</li>
                            <li>• Improved agent collaboration</li>
                            <li>• Enhanced learning algorithms</li>
                        </ul>
                    </div>
                    <div>
                        <h4 class="font-semibold text-gray-900 mb-2">Clinical Applications</h4>
                        <ul class="text-sm text-gray-600 space-y-1">
                            <li>• Real-world validation studies</li>
                            <li>• Healthcare system integration</li>
                            <li>• Patient outcome evaluation</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Call to Action -->
    <section class="py-16 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-blue-600 to-purple-700">
        <div class="max-w-4xl mx-auto text-center">
            <h2 class="text-3xl md:text-4xl font-bold text-white mb-6">
                Join Our Research Journey
            </h2>
            <p class="text-xl text-blue-100 mb-8">
                Explore our research, contribute to the project, or collaborate with our team 
                to advance AI in precision medicine
            </p>
            <div class="flex flex-col sm:flex-row gap-4 justify-center">
                <a href="http://pitools.niper.ac.in/biomedkai/home" target="_blank" 
                   class="bg-white text-blue-600 px-8 py-4 rounded-xl font-semibold hover:bg-gray-100 transition-colors inline-flex items-center justify-center">
                    <i class="fas fa-external-link-alt mr-3"></i>
                    Explore Platform
                </a>
                <a href="https://github.com/chemplusx/BioMed-KAI" target="_blank" 
                   class="border-2 border-white text-white px-8 py-4 rounded-xl font-semibold hover:bg-white hover:text-blue-600 transition-colors inline-flex items-center justify-center">
                    <i class="fab fa-github mr-3"></i>
                    View Research Code
                </a>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer class="bg-white py-12 px-4 sm:px-6 lg:px-8">
        <div class="max-w-7xl mx-auto text-center">
            <div class="flex items-center justify-center space-x-3 mb-4">
                <div class="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                    <i class="fas fa-brain text-white"></i>
                </div>
                <div>
                    <h3 class="text-lg font-bold text-gray-900">BioMed-KAI Research</h3>
                    <p class="text-sm text-gray-600">Advancing Precision Medicine</p>
                </div>
            </div>
            <p class="text-gray-600 mb-4">
                Pioneering agentic AI for healthcare through collaborative research and innovation
            </p>
            <div class="flex justify-center space-x-4">
                <a href="index.html" class="text-gray-600 hover:text-blue-600 transition-colors">Home</a>
                <a href="sources.html" class="text-gray-600 hover:text-blue-600 transition-colors">Sources</a>
                <a href="https://github.com/chemplusx/BioMed-KAI" target="_blank" class="text-gray-600 hover:text-blue-600 transition-colors">GitHub</a>
            </div>
        </div>
    </footer>

    <script>
        // Mobile menu toggle
        document.getElementById('mobile-menu-button').addEventListener('click', function() {
            const menu = document.getElementById('mobile-menu');
            menu.classList.toggle('hidden');
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
