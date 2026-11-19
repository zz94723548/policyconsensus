"""Declared episode roles for the two frozen-router training protocols."""


def router_cache_roles(identity):
    partition=identity.get('collection_partition','risk_train')
    if partition=='risk_train':
        return identity['risk_split']['splits']
    if partition!='expert_router':
        raise ValueError('Router cache must contain risk_train or expert_router data')
    original=identity['source_manifest']
    roles={name:original['splits'][name] for name in ['expert_train','expert_val']}
    all_ids=roles['expert_train']+roles['expert_val']
    if len(set(all_ids))!=len(all_ids):
        raise ValueError('Expert router training/validation overlap')
    for role,count in [('expert_train',30),('expert_val',5)]:
        for task in original['task_order']:
            if sum(original['task_by_episode'][i]==task for i in roles[role])!=count:
                raise ValueError('Expert router requires the original per-task 30/5 split')
    for role in ['risk_train','risk_calibration','test']:
        if set(all_ids)&set(original['splits'][role]):
            raise ValueError('Expert router overlaps reserved data')
    return roles
