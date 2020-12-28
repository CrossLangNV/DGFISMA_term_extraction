from typing import List


class ConceptsVoc(dict):
    def get_all_terms(self) -> List[str]:
        """ get all unique terms found in the different concepts.
        One concept can consist of multiple terms and the same term can be used in multiple concepts.

        Returns:
            List of strings with all the terms.
        """
        return list(set(term_j for l_terms_i in self.values() for term_j in l_terms_i))
