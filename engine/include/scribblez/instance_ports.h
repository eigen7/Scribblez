#pragma once

namespace scribblez {

// Amount to add to a default network port so parallel dev-container instances
// don't collide. Read from the DEVENV_INSTANCE_PORT_OFFSET environment variable
// (set by run_docker.py for instance N to instance_port_stride * N); 0 when the
// variable is unset, empty, or not a non-negative integer.
//
// Apply it only to *default* ports: an explicitly requested port (e.g. --port)
// is used verbatim.
int instance_port_offset();

}  // namespace scribblez
