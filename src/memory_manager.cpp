namespace pictor {

memory_manager::memory_manager(context &ctx, size_t total_device_memory_limit,
                               float data_pool_ratio)
    : _main_resource(ctx) {

  // Split the total limit
  size_t data_size =
      static_cast<size_t>(total_device_memory_limit * data_pool_ratio);
  size_t workspace_size = total_device_memory_limit - data_size;

  // Create pools
  _data_pool = std::make_unique<pool_memory_resource>(_main_resource, data_size);
  _workspace_pool = std::make_unique<pool_memory_resource>(_main_resource, workspace_size);
}

std::pair<size_t, size_t> memory_manager::get_data_memory_usage() {
  // Stats not dynamically available in simple bump allocator without extra tracking
  return {0, 0};
}

void memory_manager::reset_workspace() {
  if (_workspace_pool) {
    _workspace_pool->reset();
  }
}

std::pair<size_t, size_t> memory_manager::get_workspace_memory_usage() {
  return {0, 0};
}

std::pair<size_t, size_t> memory_manager::get_total_memory_usage() {
  return {0, 0};
}

} // namespace rasterdf
