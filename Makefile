MANIFEST_FILES = manifest.json requirements.txt
SOURCE_FILES := src/reedfrost/__init__.py src/reedfrost/app.py

.PHONY: local deploy clean

local:
	poetry run streamlit run src/reedfrost/app.py

deploy: $(MANIFEST_FILES) $(SOURCE_FILES)
	rsconnect deploy \
		manifest manifest.json \
		--title reedfrost

manifest.json requirements.txt: $(SOURCE_FILES)
	rm -f requirements.txt
	rsconnect write-manifest streamlit . \
		$(SOURCE_FILES) \
		--exclude "**" \
		--entrypoint src/reedfrost/app.py \
		--overwrite

clean:
	rm -f $(MANIFEST_FILES)
