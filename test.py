def pad_sents(sents, pad_token):
	""" Pad list of sentences according to the longest sentence in the batch.
	@param sents (list[list[str]]): list of sentences, where each sentence
									is represented as a list of words
	@param pad_token (str): padding token
	@returns sents_padded (list[list[str]]): list of sentences where sentences shorter
		than the max length sentence are padded out with the pad_token, such that
		each sentences in the batch now has equal length.
	"""
	sents_padded = []

	### YOUR CODE HERE (~6 Lines)
	max_val = max([len(sent) for sent in sents])
	for sentence in sents:
		if len(sentence) < max_val:
			while len(sentence) < max_val:
				sentence.append(pad_token)
		sents_padded.append(sentence)

	### END YOUR CODE

	return sents_padded


