"""Immutable source identity for the reviewed official Cosmos policy runtime."""

AUDITED_COSMOS_REPO_COMMIT = "583ba1d85b51148e898bdd4232cfeee73f6fce1e"
AUDITED_COSMOS_SOURCE_SHA256 = (
    "c8cf94e18f840dda55afa162d6f6b0a4cada36fbbd8bbb45bf27c131f940f980"
)


def has_exact_audited_cosmos_source(*, repo_commit, source_sha256) -> bool:
    return (
        repo_commit == AUDITED_COSMOS_REPO_COMMIT
        and source_sha256 == AUDITED_COSMOS_SOURCE_SHA256
    )
