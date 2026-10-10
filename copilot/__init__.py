"""Job Application Copilot.

Layering (dependencies point inward only):

    api / graph / mcp_servers      -> delivery
    services                       -> use cases, depend on ports
    ports                          -> interfaces (typing.Protocol)
    domain                         -> pure models and rules
    adapters                       -> implement ports with real libraries
    container                      -> composition root, wires adapters into services
"""

__version__ = "0.1.0"
