.PHONY: install clean 

install:
	pip install -r requirements.txt

clean:
	rm -rf __pycache__ .pytest_cache

EC2_HOST = sunking-ec2
REMOTE_DIR = /home/ubuntu/cbr360_plus/
LOCAL_DIR = $(PWD)/

pull-ec2:
	sh /Users/kiprono/Documents/70_System/mac_setup/backup_to_ssd.sh
	rsync -avz \
	--exclude='.venv' \
	--exclude="data" \
	--exclude="output" \
	$(EC2_HOST):$(REMOTE_DIR) $(LOCAL_DIR)

push-ec2:
	rsync -avz \
	--exclude='.venv' \
	--exclude=".DS_Store" \
	--exclude=".git/" \
	$(LOCAL_DIR) $(EC2_HOST):$(REMOTE_DIR)

# Dry runs
pull-ec2-dry:
	rsync -avzn \
	--exclude='.venv' \
	--exclude="data" \
	--exclude="output" \
	$(EC2_HOST):$(REMOTE_DIR) $(LOCAL_DIR)

push-ec2-dry:
	rsync -avzn \
	--exclude='.venv' \
	--exclude=".DS_Store" \
	--exclude=".git/" \
	$(LOCAL_DIR) $(EC2_HOST):$(REMOTE_DIR)
