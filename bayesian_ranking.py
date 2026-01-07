import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.options.display.width = 0

def pseudo_bernoulli():
    '''
    Considers ratings to be a sort of bernoulli distribution. Each rating is converted to an independent sample of 10
    trials. For example 5.0 -> 10 likes out of 10 views, 2.5 -> 5 likes out of 10 views. Then we use the number of
    likes to update our prior beta distribution. Note that this creates new information out of nowhere and relies on the
    assumption that these ratings actually correspond to such a distribution, which is likely wrong. However, we will
    see that this still leads to some decent results, as long as we filter movies with very few ratings.
    '''
    ratings = pd.read_csv("movielens20/rating.csv")
    ratings = ratings.drop("timestamp", axis=1)
    movies = pd.read_csv("movielens20/movie.csv")

    ratings.merge(movies[["movieId", "title"]], on="movieId")
    averages = ratings.groupby("movieId")["rating"].mean()
    # number of ratings for each movie
    counts = ratings.groupby("movieId")["rating"].count()
    success = 2*counts*averages
    fails = 10*counts - success

    # flat beta distributions, i.e. uniform distribution
    betas = movies.copy()
    betas = betas.drop("genres", axis=1)
    betas.set_index("movieId", inplace=True)
    betas["alpha"] = 1 + success
    betas["beta"] = 1 + 10*counts - success

    betas["expected"] = betas["alpha"]/(betas["alpha"]+betas["beta"])
    counts = counts.reindex(betas.index, fill_value=0)
    print(betas[counts>100].sort_values("expected", ascending=False).head(50))

def bernoulli():
    '''
    Translates ratings into either likes or dislikes. Then each we consider ratings to be a bernoulli distribution with
    a flat beta prior. Choosing a threshold for what counts as a like or dislike is hard, this is because if ratings
    are optimistic, then a 3 might mean a bad movie but a threshold of 2.5 would view that as a like. We will use a
    threshold of 2.5 regardless for now. So ratings higher than 2.5 will be viewed as likes
    '''
    ratings = pd.read_csv("movielens20/rating.csv")
    ratings = ratings.drop("timestamp", axis=1)
    movies = pd.read_csv("movielens20/movie.csv")
    ratings.merge(movies[["movieId", "title"]], on="movieId")

    val_counts = ratings.groupby("movieId")["rating"].value_counts()
    likes = ratings[ratings["rating"]>2.5].groupby("movieId")["rating"].count()
    total_ratings = ratings.groupby("movieId")["rating"].count()

    # beta distributions with flat prior
    betas = movies.copy()
    betas = betas.drop("genres", axis=1)
    betas.set_index("movieId", inplace=True)
    betas["alpha"] = 1 + likes
    betas["beta"] = 1 + total_ratings - likes

    betas["expected"] = betas["alpha"]/(betas["alpha"]+betas["beta"])
    total_ratings = total_ratings.reindex(betas.index)
    print(betas[total_ratings>100].sort_values("expected", ascending=False).head(50))

def multinomial():
    '''
    Uses a multinomial distribution as the distribution of the ratings so each rating has a probability of being chosen
    by users. We use a dirichlet conjugate prior in a similar fashion. Note that the multinomial and dirichlet
    distributions are just the multivariate versions of the binomial and beta distributions.
    '''
    ratings = pd.read_csv("movielens20/rating.csv")
    ratings = ratings.drop("timestamp", axis=1)
    movies = pd.read_csv("movielens20/movie.csv")
    ratings.merge(movies[["movieId", "title"]], on="movieId")

    val_counts = ratings.groupby("movieId")["rating"].value_counts().reset_index()
    val_counts = val_counts.pivot(index="movieId", columns="rating", values="count").fillna(0)
    total_ratings = ratings.groupby("movieId")["rating"].count()

    dirichlet = movies.copy()
    dirichlet = dirichlet.drop("genres", axis=1)
    dirichlet.set_index("movieId", inplace=True)

    for i in range(10):
        dirichlet["alpha"+str(i+1)] = 1 + val_counts[(i+1)/2]

    dirichlet["total_ratings"] = total_ratings
    dirichlet["expected"] = 0
    for i in range(10):
        dirichlet["expected"] = dirichlet["expected"] + 0.5*(i+1)*dirichlet["alpha"+str(i+1)]/(total_ratings+10)

    print(dirichlet.sort_values("expected", ascending=False).head(50))






#pseudo_bernoulli()
#bernoulli()
multinomial()
