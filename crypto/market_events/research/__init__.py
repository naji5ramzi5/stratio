"""Research data foundation (Phase 2).

Feature research is built ON TOP of the stable Phase 1 infrastructure:
- data comes exclusively from the Phase 1 event store (never re-parsed zips)
- PIT discipline is enforced by construction + automated assertions
- feature generation and label generation are separate engines
No Phase 1 module is modified or re-implemented here.
"""
