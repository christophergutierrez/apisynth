"""
Shared constants and utilities for sweep.py and run.py.
"""

# Params excluded from variant dimensions and filter cycling.
# These are sort, pagination, meta, or free-text params that don't form meaningful
# combinatorial variants for training data.
_SKIP_FILTER = frozenset({
    "pageSize", "pageToken", "orderBy", "sorts", "q", "name", "query",
    "createdAt", "requestId", "mediaGroupName", "reportingScope",
    "programIds", "networkIds", "audienceIds", "audienceUuids",
    "filteredAgencyIds", "filteredAgencyAdvertiserIds",
    "useCases", "year", "level", "cadences", "fetchRecipientAncestorPath",
})

# sweep.py uses the same set but without pageToken (sweep considers pageToken a variant dim)
_SKIP_VARIANT = _SKIP_FILTER - {"pageToken"}

PAGE_SIZES = [1, 3, 5, 10, 20, 25, 50, 100, 200, 500, 1000]


def humanize(s: str) -> str:
    return s.replace("-", " ").replace("_", " ")


def singular(s: str) -> str:
    if s.endswith("ies"):
        return s[:-3] + "y"
    if s.endswith("ses"):
        return s[:-2]
    if s.endswith("s") and not s.endswith("ss"):
        return s[:-1]
    return s
