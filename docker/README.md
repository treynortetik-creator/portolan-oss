# Docker

Docker runs through Colima on the dev Mac (`colima start`). `scripts/ci_local.py` builds this image, runs the suite in it and checks the web role before every push; the image is arm64 here while Railway builds amd64.
