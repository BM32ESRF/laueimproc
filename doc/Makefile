# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = laueimproc
SOURCEDIR     = .
BUILDDIR      = build
TEMPLATEDIR   = template

clean:
	@echo "****************************************"
	@echo "************* Remove Build *************"
	@echo "****************************************"
	@rm -rf "$(BUILDDIR)"

html:
	@echo "****************************************"
	@echo "************** Invoke Pdoc *************"
	@echo "****************************************"
	@pdoc --html --force --template-dir "$(TEMPLATEDIR)" --output-dir "$(BUILDDIR)" "../$(SPHINXPROJ)"

	@echo "****************************************"
	@echo "************* Invoke Sphinx ************"
	@echo "****************************************"
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
