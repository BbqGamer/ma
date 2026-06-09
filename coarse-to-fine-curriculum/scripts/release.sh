#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-bbqdocker/coarse-to-fine-curriculum}"
PLATFORM="${PLATFORM:-linux/amd64}"
VERSION="${VERSION:-$(cat VERSION)}"
GIT_SHA="$(git rev-parse --short HEAD)"
TAG="v${VERSION}-${GIT_SHA}"

echo "Releasing:"
echo "  ${IMAGE}:${TAG}"
echo "  ${IMAGE}:latest"

docker buildx build \
  --platform "${PLATFORM}" \
  -t "${IMAGE}:${TAG}" \
  -t "${IMAGE}:latest" \
  --push \
  .

echo
echo "Done."
echo "Pushed ${IMAGE}:${TAG} and ${IMAGE}:latest"
