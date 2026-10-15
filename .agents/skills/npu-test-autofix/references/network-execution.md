# Network Execution

Use the network configuration already provided by the company environment.
Company-internal model APIs, package mirrors, artifact stores, Git services,
and dataset/model repositories are allowed.

Do not force offline environment variables. Preserve existing settings such as
`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, certificate bundles, and internal
endpoint variables. Never print API keys, tokens, passwords, or full signed
URLs into logs or reports.

Prefer resources in this order:

1. Existing local weights, datasets, converted checkpoints, and installed packages.
2. Company-internal model APIs, mirrors, repositories, and artifact services.
3. Public internet resources only when reachable and appropriate for the company environment.

The model executing this skill may use the configured company-internal LLM API
for diagnosis and code iteration. Scripts in this skill remain deterministic
and must not hardcode a model vendor, endpoint, credential, or public mirror.

When a download or dependency installation is necessary, use the environment's
configured source. Record the command and resulting artifact without exposing
credentials. A network failure is `ENVIRONMENT` or `DEPENDENCY` evidence,
not automatically a product-code defect.
