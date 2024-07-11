def group_scores(test_authors, avg_preds, args):
    score, score_partner, score_rest = 0, 0, 0
    n_test = len(test_authors)
    # Calculate the scores for either topic-controlled of conversational corpus
    if args.corpus_name == 'Frida' or args.corpus_name == "RFM":
        for j in range(len(test_authors)):
            if test_authors[j] == avg_preds[j]:
                score += 1
            elif test_authors[j] == avg_preds[j] - 1 and test_authors[j] % 2 == 1:
                score_partner += 1
            elif test_authors[j] == avg_preds[j] + 1 and test_authors[j] % 2 == 0:
                score_partner += 1
            else:
                score_rest += 1
    elif args.corpus_name == 'abc_nl1':
        for j in range(len(test_authors)):
            if test_authors[j] == avg_preds[j]:
                score += 1
            elif avg_preds[j] % 2 == 0 and test_authors[j] % 2 == 1:
                score_partner += 1
            elif avg_preds[j] % 2 == 1 and test_authors[j] % 2 == 0:
                score_partner += 1
            else:
                score_rest += 1

    return score/n_test, score_partner/n_test, score_rest/n_test