MANIFEST_FILES = manifest.json requirements.txt
SOURCE_FILES := src/reedfrost/__init__.py src/reedfrost/app/__init__.py src/reedfrost/app/input.py src/reedfrost/app/model.py src/reedfrost/app/view.py
ENTRYPOINT := src/reedfrost/app/__init__.py

.PHONY: local deploy clean docs

local:
	uv run streamlit run $(ENTRYPOINT)

deploy: $(MANIFEST_FILES) $(SOURCE_FILES)
	rsconnect deploy \
		manifest manifest.json \
		--title reedfrost

manifest.json requirements.txt: $(SOURCE_FILES)
	rm -f requirements.txt
	rsconnect write-manifest streamlit . \
		$(SOURCE_FILES) \
		--exclude "**" \
		--entrypoint $(ENTRYPOINT) \
		--overwrite

clean:
	rm -f $(MANIFEST_FILES)

docs:
	uv run mkdocs serve
