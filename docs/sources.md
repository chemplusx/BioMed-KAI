---
layout: default
title: Sources
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
                    <a href="sources.html" class="nav-active px-3 py-2 text-sm font-medium transition-colors">Sources</a>
                    <a href="about.html" class="text-gray-600 hover:text-blue-600 px-3 py-2 text-sm font-medium transition-colors">About</a>
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
                    <a href="sources.html" class="block px-3 py-2 text-blue-600 font-medium">Sources</a>
                    <a href="about.html" class="block px-3 py-2 text-gray-600 hover:text-blue-600">About</a>
                    <a href="https://github.com/chemplusx/BioMed-KAI" target="_blank" class="block px-3 py-2 text-gray-600 hover:text-blue-600">
                        <i class="fab fa-github mr-1"></i>GitHub
                    </a>
                </div>
            </div>
        </div>
    </nav>

    <!-- Header -->
    <section class="pt-24 pb-16 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-blue-50 to-purple-50">
        <div class="max-w-7xl mx-auto text-center">
            <h1 class="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
                Knowledge Sources
            </h1>
            <p class="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
                Explore the comprehensive collection of databases and ontologies that power BioMed-KAI's knowledge graph
            </p>
            
            <!-- Search and Filter -->
            <div class="max-w-2xl mx-auto mb-8">
                <div class="relative">
                    <input type="text" id="search-input" placeholder="Search sources..." 
                           class="search-box w-full px-4 py-3 pl-12 rounded-xl focus:outline-none">
                    <i class="fas fa-search absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400"></i>
                </div>
            </div>

            <!-- Filter Buttons -->
            <div class="flex flex-wrap justify-center gap-3 mb-8">
                <button class="filter-btn active px-4 py-2 rounded-full text-sm font-medium" data-filter="all">
                    All Sources
                </button>
                <button class="filter-btn px-4 py-2 rounded-full text-sm font-medium" data-filter="database">
                    Databases
                </button>
                <button class="filter-btn px-4 py-2 rounded-full text-sm font-medium" data-filter="ontology">
                    Ontologies
                </button>
            </div>

            <!-- Stats -->
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-2xl mx-auto">
                <div class="bg-white rounded-xl p-4 shadow-sm">
                    <div class="text-2xl font-bold text-blue-600">21</div>
                    <div class="text-sm text-gray-600">Databases</div>
                </div>
                <div class="bg-white rounded-xl p-4 shadow-sm">
                    <div class="text-2xl font-bold text-purple-600">15</div>
                    <div class="text-sm text-gray-600">Ontologies</div>
                </div>
                <div class="bg-white rounded-xl p-4 shadow-sm">
                    <div class="text-2xl font-bold text-green-600">36</div>
                    <div class="text-sm text-gray-600">Total Sources</div>
                </div>
                <div class="bg-white rounded-xl p-4 shadow-sm">
                    <div class="text-2xl font-bold text-orange-600">200M+</div>
                    <div class="text-sm text-gray-600">Relationships</div>
                </div>
            </div>
        </div>
    </section>

    <!-- Databases Section -->
    <section class="py-16 px-4 sm:px-6 lg:px-8" id="databases-section">
        <div class="max-w-7xl mx-auto">
            <h2 class="text-3xl font-bold text-gray-900 mb-8 text-center">
                <i class="fas fa-database text-blue-600 mr-3"></i>
                Integrated Databases (21)
            </h2>
            
            <div id="databases-grid" class="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                <!-- Database cards will be populated by JavaScript -->
            </div>
        </div>
    </section>

    <!-- Ontologies Section -->
    <section class="py-16 px-4 sm:px-6 lg:px-8 bg-gray-50" id="ontologies-section">
        <div class="max-w-7xl mx-auto">
            <h2 class="text-3xl font-bold text-gray-900 mb-8 text-center">
                <i class="fas fa-project-diagram text-purple-600 mr-3"></i>
                Medical Ontologies (15)
            </h2>
            
            <div id="ontologies-grid" class="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                <!-- Ontology cards will be populated by JavaScript -->
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
                    <h3 class="text-lg font-bold text-gray-900">BioMed-KAI</h3>
                    <p class="text-sm text-gray-600">Knowledge Sources</p>
                </div>
            </div>
            <p class="text-gray-600 mb-4">
                Comprehensive biomedical knowledge integration for precision medicine
            </p>
            <div class="flex justify-center space-x-4">
                <a href="index.html" class="text-gray-600 hover:text-blue-600 transition-colors">Home</a>
                <a href="about.html" class="text-gray-600 hover:text-blue-600 transition-colors">About</a>
                <a href="https://github.com/chemplusx/BioMed-KAI" target="_blank" class="text-gray-600 hover:text-blue-600 transition-colors">GitHub</a>
            </div>
        </div>
    </footer>

    <script>
        // Data sources
        const databases = [
            { name: "UniProt", description: "Universal Protein Resource", version: "v07-2024", type: "Protein", category: "database", icon: "fas fa-dna", url: "https://ftp.uniprot.org/pub/databases/uniprot/current_release/" },
            { name: "DrugBank", description: "Drug and Drug Target Database", version: "5.1.12", type: "Drug", category: "database", icon: "fas fa-capsules", url: "https://drugbank.ca" },
            { name: "OpenTargets", description: "Target Identification Platform", version: "24.06", type: "Genomics", category: "database", icon: "fas fa-bullseye", url: "http://ftp.ebi.ac.uk/pub/databases/opentargets/platform" },
            { name: "Ensembl", description: "Genome Annotation Database", version: "112", type: "Genomics", category: "database", icon: "fas fa-dna", url: "https://ftp.ensembl.org/pub/current_genbank/" },
            { name: "HGNC", description: "HUGO Gene Nomenclature Committee", version: "v2024-08-23", type: "Genomics", category: "database", icon: "fas fa-gene", url: "https://www.genenames.org/download/statistics-and-files/" },
            { name: "ChEBI", description: "Chemical Entities of Biological Interest", version: "v08-2024", type: "Chemical", category: "database", icon: "fas fa-flask", url: "https://www.ebi.ac.uk/chebi/downloadsForward.do" },
            { name: "IntAct", description: "Protein Interaction Database", version: "v2024-05-20", type: "Protein", category: "database", icon: "fas fa-handshake", url: "https://www.ebi.ac.uk/intact/download/ftp" },
            { name: "TTD", description: "Therapeutic Target Database", version: "10.1.01", type: "Drug", category: "database", icon: "fas fa-bullseye", url: "https://idrblab.net/ttd/full-data-download" },
            { name: "MarkerDB", description: "Biomarker Database", version: "2", type: "Genomics", category: "database", icon: "fas fa-chart-bar", url: "https://markerdb.ca/downloads" },
            { name: "DrugCentral", description: "Drug Information Resource", version: "v2023-05", type: "Drug", category: "database", icon: "fas fa-pills", url: "https://drugcentral.org/ActiveDownload" },
            { name: "GWAS Catalog", description: "Genome-Wide Association Studies", version: "V1.0.3.1", type: "Studies", category: "database", icon: "fas fa-chart-line", url: "https://www.ebi.ac.uk/gwas/docs/file-downloads" },
            { name: "HPA", description: "Human Protein Atlas", version: "23", type: "Protein", category: "database", icon: "fas fa-microscope", url: "https://www.proteinatlas.org/about/download" },
            { name: "Reactome", description: "Pathway Database", version: "v89", type: "Pathways", category: "database", icon: "fas fa-sitemap", url: "https://reactome.org/download-data" },
            { name: "dbSNP", description: "Single Nucleotide Polymorphisms", version: "v2023-01-25", type: "Genomics", category: "database", icon: "fas fa-dna", url: "https://ftp.ncbi.nih.gov/snp/" },
            { name: "HMDB", description: "Human Metabolome Database", version: "v5", type: "Metabolomics", category: "database", icon: "fas fa-atom", url: "https://hmdb.ca/downloads" },
            { name: "PubMed", description: "Biomedical Literature Database", version: "v2023-12-15", type: "Studies", category: "database", icon: "fas fa-book-medical", url: "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/" },
            { name: "EMBL-EBI", description: "European Bioinformatics Institute", version: "v2024_04", type: "Database", category: "database", icon: "fas fa-database", url: "https://www.ebi.ac.uk/services" },
            { name: "BioStudies", description: "Biological Studies Database", version: "v2024-08", type: "Studies", category: "database", icon: "fas fa-search", url: "https://www.ebi.ac.uk/biostudies/api/v1/search" },
            { name: "PDB", description: "Protein Data Bank", version: "v2024-08-30", type: "Protein", category: "database", icon: "fas fa-cube", url: "ftp://ftp.wwpdb.org/" },
            { name: "NCBI", description: "National Center for Biotechnology Information", version: "v2023-12-07", type: "Database", category: "database", icon: "fas fa-database", url: "https://ftp.ncbi.nlm.nih.gov/" },
            { name: "Uberon", description: "Uber-anatomy Ontology", version: "2024-09-03", type: "Anatomy", category: "database", icon: "fas fa-brain", url: "https://obophenotype.github.io/uberon/current_release/" }
        ];

        const ontologies = [
            { name: "MonDO", description: "Monarch Disease Ontology", version: "v2024-09-03", type: "Disease", category: "ontology", icon: "fas fa-procedures", url: "https://monarchinitiative.org/ontology/mondo.owl" },
            { name: "Uberon", description: "Uber-anatomy Ontology", version: "2024-09-03", type: "Anatomy", category: "ontology", icon: "fas fa-brain", url: "https://obophenotype.github.io/uberon/current_release/" },
            { name: "PO", description: "Protein Ontology", version: "v2024-08-30", type: "Protein", category: "ontology", icon: "fas fa-dna", url: "http://purl.obolibrary.org/obo/po.owl" },
            { name: "GO", description: "Gene Ontology", version: "v2024-09-01", type: "Gene", category: "ontology", icon: "fas fa-gene", url: "http://purl.obolibrary.org/obo/go.owl" },
            { name: "HPO", description: "Human Phenotype Ontology", version: "v2024-09-01", type: "Phenotype", category: "ontology", icon: "fas fa-user-md", url: "http://purl.obolibrary.org/obo/hp.owl" },
            { name: "EFO", description: "Experimental Factor Ontology", version: "v2024-09-01", type: "Experimental Factor", category: "ontology", icon: "fas fa-flask", url: "http://purl.obolibrary.org/obo/efo.owl" },
            { name: "ChEBI", description: "Chemical Entities Ontology", version: "v2024-09-01", type: "Chemical", category: "ontology", icon: "fas fa-flask", url: "http://purl.obolibrary.org/obo/chebi.owl" },
            { name: "HDO", description: "Human Disease Ontology", version: "v2024-09-01", type: "Disease", category: "ontology", icon: "fas fa-procedures", url: "http://purl.obolibrary.org/obo/doid.owl" },
            { name: "SNOMED CT", description: "Clinical Terms", version: "v2024-09-01", type: "Clinical", category: "ontology", icon: "fas fa-stethoscope", url: "http://purl.obolibrary.org/obo/snomed.owl" },
            { name: "FoodOn", description: "Food Ontology", version: "v2024-09-01", type: "Food", category: "ontology", icon: "fas fa-apple-alt", url: "http://purl.obolibrary.org/obo/foodon.owl" },
            { name: "SYMP", description: "Symptom Ontology", version: "v2024-09-01", type: "Symptom", category: "ontology", icon: "fas fa-thermometer-half", url: "http://purl.obolibrary.org/obo/symp.owl" },
            { name: "Orphanet", description: "Rare Disease Ontology", version: "v2024-09-01", type: "Disease", category: "ontology", icon: "fas fa-procedures", url: "http://purl.obolibrary.org/obo/orphanet.owl" },
            { name: "OMIM", description: "Online Mendelian Inheritance", version: "v2024-09-01", type: "Disease", category: "ontology", icon: "fas fa-dna", url: "http://purl.obolibrary.org/obo/omim.owl" },
            { name: "CL", description: "Cell Ontology", version: "v2024-09-01", type: "Cell", category: "ontology", icon: "fas fa-circle", url: "http://purl.obolibrary.org/obo/cl.owl" },
            { name: "PATO", description: "Phenotype Attribute Ontology", version: "v2024-09-01", type: "Phenotype", category: "ontology", icon: "fas fa-eye", url: "http://purl.obolibrary.org/obo/pato.owl" }
        ];

        function getTypeColor(type) {
            const colors = {
                'Protein': 'bg-blue-100 text-blue-800',
                'Drug': 'bg-green-100 text-green-800',
                'Genomics': 'bg-purple-100 text-purple-800',
                'Chemical': 'bg-yellow-100 text-yellow-800',
                'Studies': 'bg-red-100 text-red-800',
                'Database': 'bg-gray-100 text-gray-800',
                'Pathways': 'bg-indigo-100 text-indigo-800',
                'Metabolomics': 'bg-pink-100 text-pink-800',
                'Anatomy': 'bg-orange-100 text-orange-800',
                'Disease': 'bg-red-100 text-red-800',
                'Gene': 'bg-purple-100 text-purple-800',
                'Phenotype': 'bg-blue-100 text-blue-800',
                'Experimental Factor': 'bg-green-100 text-green-800',
                'Clinical': 'bg-teal-100 text-teal-800',
                'Food': 'bg-yellow-100 text-yellow-800',
                'Symptom': 'bg-red-100 text-red-800',
                'Cell': 'bg-indigo-100 text-indigo-800'
            };
            return colors[type] || 'bg-gray-100 text-gray-800';
        }

        function createSourceCard(source) {
            return `
                <div class="source-card rounded-2xl p-6 source-item" data-category="${source.category}" data-name="${source.name.toLowerCase()}" data-description="${source.description.toLowerCase()}">
                    <div class="flex items-start justify-between mb-4">
                        <div class="icon-bg w-12 h-12 rounded-xl flex items-center justify-center">
                            <i class="${source.icon} text-xl"></i>
                        </div>
                        <span class="category-tag">${source.category === 'database' ? 'Database' : 'Ontology'}</span>
                    </div>
                    
                    <h3 class="text-xl font-bold text-gray-900 mb-2">${source.name}</h3>
                    <p class="text-gray-600 mb-3">${source.description}</p>
                    
                    <div class="flex items-center justify-between mb-4">
                        <span class="text-sm text-gray-500">Version: ${source.version}</span>
                        <span class="px-3 py-1 rounded-full text-xs font-medium ${getTypeColor(source.type)}">${source.type}</span>
                    </div>
                    
                    <a href="${source.url}" target="_blank" class="inline-flex items-center text-blue-600 hover:text-blue-800 font-medium transition-colors">
                        <i class="fas fa-external-link-alt mr-2 text-sm"></i>
                        Access Source
                    </a>
                </div>
            `;
        }

        // Populate grids
        document.getElementById('databases-grid').innerHTML = databases.map(createSourceCard).join('');
        document.getElementById('ontologies-grid').innerHTML = ontologies.map(createSourceCard).join('');

        // Filter functionality
        const filterButtons = document.querySelectorAll('.filter-btn');
        const sourceItems = document.querySelectorAll('.source-item');
        const searchInput = document.getElementById('search-input');

        filterButtons.forEach(button => {
            button.addEventListener('click', () => {
                // Update active button
                filterButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                
                const filter = button.getAttribute('data-filter');
                filterSources(filter, searchInput.value);
            });
        });

        // Search functionality
        searchInput.addEventListener('input', (e) => {
            const activeFilter = document.querySelector('.filter-btn.active').getAttribute('data-filter');
            filterSources(activeFilter, e.target.value);
        });

        function filterSources(category, searchTerm) {
            sourceItems.forEach(item => {
                const itemCategory = item.getAttribute('data-category');
                const itemName = item.getAttribute('data-name');
                const itemDescription = item.getAttribute('data-description');
                
                const matchesCategory = category === 'all' || itemCategory === category;
                const matchesSearch = !searchTerm || 
                                    itemName.includes(searchTerm.toLowerCase()) || 
                                    itemDescription.includes(searchTerm.toLowerCase());
                
                if (matchesCategory && matchesSearch) {
                    item.style.display = 'block';
                    item.style.animation = 'fadeInUp 0.5s ease forwards';
                } else {
                    item.style.display = 'none';
                }
            });

            // Show/hide sections based on visible items
            updateSectionVisibility();
        }

        function updateSectionVisibility() {
            const visibleDatabases = document.querySelectorAll('#databases-grid .source-item[style*="block"]').length;
            const visibleOntologies = document.querySelectorAll('#ontologies-grid .source-item[style*="block"]').length;
            
            document.getElementById('databases-section').style.display = visibleDatabases > 0 ? 'block' : 'none';
            document.getElementById('ontologies-section').style.display = visibleOntologies > 0 ? 'block' : 'none';
        }

        // Mobile menu toggle
        document.getElementById('mobile-menu-button').addEventListener('click', function() {
            const menu = document.getElementById('mobile-menu');
            menu.classList.toggle('hidden');
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
