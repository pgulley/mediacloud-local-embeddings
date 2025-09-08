from EmbeddingContext import LocalEmbeddingContext

test_query = '''
(police OR victim* OR crim* OR prison OR arrest* OR suspect) AND ("gun control" OR "gun restrict"~5 OR "gun restriction"~5 OR "gun restrictions"~5 OR "gun law" OR "gun laws")
'''
test_window = 30 #Search for stories from the last 30 days

if __name__ == "__main__":
	lec = LocalEmbeddingContext(test_query, test_window)

	matches = lec.search("stories about minnesota")
	print(matches)