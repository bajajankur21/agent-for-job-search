from . import greenhouse, lever, ashby, workable

FETCHERS = {
    "greenhouse": greenhouse.fetch_jobs,
    "lever": lever.fetch_jobs,
    "ashby": ashby.fetch_jobs,
    "workable": workable.fetch_jobs,
}
