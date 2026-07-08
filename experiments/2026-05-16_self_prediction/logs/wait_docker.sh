for i in $(seq 1 72); do
  if docker info >/dev/null 2>&1; then
    echo "UBUNTU_DOCKER_READY"
    echo "swebench_images=$(docker images --format '{{.Repository}}' | grep -c sweb)"
    docker --version
    exit 0
  fi
  sleep 5
done
echo "TIMEOUT: docker not reachable from Ubuntu"
exit 1
