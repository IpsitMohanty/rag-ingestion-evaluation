from langchain_core.documents import Document

from eval.retrievers import LexicalIndex, query_hybrid


class _Store:
    def similarity_search_with_score(self, query, k):
        return [(Document(page_content="dense answer", metadata={"faq_index": 0}), 0.2)]

    class _Collection:
        def count(self):
            return 1

    _collection = _Collection()


class _Policy:
    def query_scored(self, query, k):
        return []


def test_lexical_index_returns_exact_term_match():
    index = LexicalIndex(
        [
            Document(page_content="National Food Security Act", metadata={"page": 9}),
            Document(page_content="Unrelated content", metadata={"page": 2}),
        ],
        "policy_pdf",
    )

    hits = index.query_scored("Food Security Act", 2)

    assert len(hits) == 1
    assert hits[0].page == 9


def test_hybrid_fuses_dense_and_lexical_candidates():
    lexical = LexicalIndex(
        [Document(page_content="lexical answer", metadata={"faq_index": 1})],
        "faq",
    )

    hits = query_hybrid(
        "lexical answer", 3, _Store(), _Policy(), lexical, LexicalIndex([], "policy_pdf")
    )

    assert {hit.faq_index for hit in hits} == {0, 1}
