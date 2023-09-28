from xmanager import xm
from xmanager import xm_local
from absl import app
import time
import jax
import os
from experiment_defs import *


def print_and_say(s):
    print(s)


@xm.run_in_asyncio_loop
async def main(_):
    async with xm_local.create_experiment(experiment_title='hpl_bo') as experiment:
        if IS_HPOB:
            assert False

        # construct the initial jax random key
        key = jax.random.PRNGKey(RANDOM_SEED)

        print_and_say('run config')
        
        spec = xm.PythonContainer(
            path='/home/zfan/hyperbo',
            base_image='python:3.7.11',
            entrypoint=xm.CommandList([
                'python3 hyperbo/experiments_xmanager/hpl_bo_split_config.py',
            ]),
            docker_instructions=[
                'WORKDIR bo',
                'COPY hyperbo/hyperbo hyperbo',
                'COPY hyperbo/hpob hpob',
                'COPY hyperbo/setup.py setup.py',
                'RUN python3 -m pip install --upgrade pip==23.2',
                'RUN pip3 install .',
            ],
        )

        [executable] = experiment.package([
            xm.Packageable(
                executable_spec=spec,
                executor_spec=xm_local.Vertex.Spec(),
            ),
        ])

        requirements = xm.JobRequirements(
            cpu=BASIC_CPU_COUNT,
            ram=3.9 * BASIC_CPU_COUNT * xm.GiB,
        )

        job = xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            args=None,
        )
        work_unit = await experiment.add(job)
        await work_unit.wait_until_complete()

        print_and_say('fit single GP')

        spec = xm.PythonContainer(
            path='/home/zfan/hyperbo',
            base_image='python:3.7.11',
            entrypoint=xm.CommandList([
                'python3 hyperbo/experiments_xmanager/hpl_bo_split_worker.py $@',
            ]),
            docker_instructions=[
                'WORKDIR bo',
                'COPY hyperbo/hyperbo hyperbo',
                'COPY hyperbo/hpob hpob',
                'COPY hyperbo/setup.py setup.py',
                'RUN python3 -m pip install --upgrade pip==23.2',
                'RUN pip3 install .',
            ],
        )

        [executable] = experiment.package([
            xm.Packageable(
                executable_spec=spec,
                executor_spec=xm_local.Vertex.Spec(),
            ),
        ])

        requirements = xm.JobRequirements(
            cpu=FITTING_NODE_CPU_COUNT,
            ram=3.9 * FITTING_NODE_CPU_COUNT * xm.GiB,
        )

        work_units = []

        if IS_HPOB:
            fit_single_gp_id_list = FULL_ID_LIST + ['pd1']
        else:
            fit_single_gp_id_list = FULL_ID_LIST

        for method_name in FIT_SINGLE_GP_METHOD_LIST:
            for train_id in fit_single_gp_id_list:
                new_key, key = jax.random.split(key)
                job = xm.Job(
                    executable=executable,
                    executor=xm_local.Vertex(requirements=requirements),
                    args={
                        'group_id': GROUP_ID,
                        'mode': 'fit_gp_params_setup_b_id',
                        'dataset_id': train_id,
                        'key_0': str(new_key[0]),
                        'key_1': str(new_key[1]),
                        'method_name': method_name,
                    }
                )
                work_units.append(await experiment.add(job))

        for work_unit in work_units:
            await work_unit.wait_until_complete()

        print_and_say('fit two-step HGP')

        spec = xm.PythonContainer(
            path='/home/zfan/hyperbo',
            base_image='python:3.7.11',
            entrypoint=xm.CommandList([
                'python3 hyperbo/experiments_xmanager/hpl_bo_split_worker.py $@',
            ]),
            docker_instructions=[
                'WORKDIR bo',
                'COPY hyperbo/hyperbo hyperbo',
                'COPY hyperbo/hpob hpob',
                'COPY hyperbo/setup.py setup.py',
                'RUN python3 -m pip install --upgrade pip==23.2',
                'RUN pip3 install .',
            ],
        )

        [executable] = experiment.package([
            xm.Packageable(
                executable_spec=spec,
                executor_spec=xm_local.Vertex.Spec(),
            ),
        ])

        requirements = xm.JobRequirements(
            cpu=FITTING_NODE_CPU_COUNT,
            ram=3.9 * FITTING_NODE_CPU_COUNT * xm.GiB,
        )

        work_units = []

        for num_dataset in VARY_NUM_DATASET_NUM_LIST[:3]:
            for seed in range(VARY_NUM_DATASET_NUM_SEEDS):
                new_key, key = jax.random.split(key)
                job = xm.Job(
                    executable=executable,
                    executor=xm_local.Vertex(requirements=requirements),
                    args={
                        'group_id': GROUP_ID,
                        'mode': 'fit_direct_hgp_two_step_setup_b',
                        'key_0': str(new_key[0]),
                        'key_1': str(new_key[1]),
                        'extra_suffix': '_n{}seed{}'.format(num_dataset, seed),
                    }
                )
                work_units.append(await experiment.add(job))

                new_key, key = jax.random.split(key)
                job = xm.Job(
                    executable=executable,
                    executor=xm_local.Vertex(requirements=requirements),
                    args={
                        'group_id': GROUP_ID,
                        'mode': 'fit_hpl_hgp_two_step_setup_b',
                        'key_0': str(new_key[0]),
                        'key_1': str(new_key[1]),
                        'extra_suffix': '_n{}seed{}'.format(num_dataset, seed),
                    }
                )
                work_units.append(await experiment.add(job))

        for work_unit in work_units:
            await work_unit.wait_until_complete()

        print_and_say('run BO')

        spec = xm.PythonContainer(
            path='/home/zfan/hyperbo',
            base_image='python:3.7.11',
            entrypoint=xm.CommandList([
                'python3 hyperbo/experiments_xmanager/hpl_bo_split_worker.py $@',
            ]),
            docker_instructions=[
                'WORKDIR bo',
                'COPY hyperbo/hyperbo hyperbo',
                'COPY hyperbo/hpob hpob',
                'COPY hyperbo/setup.py setup.py',
                'RUN python3 -m pip install --upgrade pip==23.2',
                'RUN pip3 install .',
            ],
        )

        [executable] = experiment.package([
            xm.Packageable(
                executable_spec=spec,
                executor_spec=xm_local.Vertex.Spec(),
            ),
        ])

        requirements = xm.JobRequirements(
            cpu=BO_NODE_CPU_COUNT,
            ram=3.9 * BO_NODE_CPU_COUNT * xm.GiB,
        )

        work_units = []

        for num_dataset in VARY_NUM_DATASET_NUM_LIST[:3]:
            for seed in range(VARY_NUM_DATASET_NUM_SEEDS):
                for ac_func_type in AC_FUNC_TYPE_LIST:
                    for test_id in TEST_ID_LIST:
                        for method_name in VARY_NUM_DATASET_METHOD_NAME_LIST:
                            new_key, key = jax.random.split(key)
                            job = xm.Job(
                                executable=executable,
                                executor=xm_local.Vertex(requirements=requirements),
                                args={
                                    'group_id': GROUP_ID,
                                    'mode': 'test_bo_setup_b_id',
                                    'dataset_id': test_id,
                                    'ac_func_type': ac_func_type,
                                    'method_name': method_name,
                                    'key_0': str(new_key[0]),
                                    'key_1': str(new_key[1]),
                                    'extra_suffix': '_n{}seed{}'.format(num_dataset, seed),
                                }
                            )
                            work_units.append(await experiment.add(job))

        for work_unit in work_units:
            await work_unit.wait_until_complete()

        print_and_say('evaluate NLL')

        spec = xm.PythonContainer(
            path='/home/zfan/hyperbo',
            base_image='python:3.7.11',
            entrypoint=xm.CommandList([
                'python3 hyperbo/experiments_xmanager/hpl_bo_split_worker.py $@',
            ]),
            docker_instructions=[
                'WORKDIR bo',
                'COPY hyperbo/hyperbo hyperbo',
                'COPY hyperbo/hpob hpob',
                'COPY hyperbo/setup.py setup.py',
                'RUN python3 -m pip install --upgrade pip==23.2',
                'RUN pip3 install .',
            ],
        )

        [executable] = experiment.package([
            xm.Packageable(
                executable_spec=spec,
                executor_spec=xm_local.Vertex.Spec(),
            ),
        ])

        requirements = xm.JobRequirements(
            cpu=NLL_NODE_CPU_COUNT,
            ram=3.9 * NLL_NODE_CPU_COUNT * xm.GiB,
        )

        work_units = []

        for num_dataset in VARY_NUM_DATASET_NUM_LIST:
            for seed in range(VARY_NUM_DATASET_NUM_SEEDS):
                for method_name in VARY_NUM_DATASET_METHOD_NAME_LIST:
                    if method_name not in EVAL_KL_METHOD_NAME_LIST:
                        continue

                    new_key, key = jax.random.split(key)
                    job = xm.Job(
                        executable=executable,
                        executor=xm_local.Vertex(requirements=requirements),
                        args={
                            'group_id': GROUP_ID,
                            'mode': 'eval_kl_with_gt',
                            'method_name': method_name,
                            'key_0': str(new_key[0]),
                            'key_1': str(new_key[1]),
                            'extra_suffix': '_n{}seed{}'.format(num_dataset, seed),
                        }
                    )
                    work_units.append(await experiment.add(job))
    
        for num_dataset in VARY_NUM_DATASET_NUM_LIST:
            for seed in range(VARY_NUM_DATASET_NUM_SEEDS):
                for dataset_id in FULL_ID_LIST:
                    for method_name in VARY_NUM_DATASET_METHOD_NAME_LIST:
                        if method_name == 'random':
                            continue
                        new_key, key = jax.random.split(key)
                        job = xm.Job(
                            executable=executable,
                            executor=xm_local.Vertex(requirements=requirements),
                            args={
                                'group_id': GROUP_ID,
                                'mode': 'eval_nll_setup_b_train_id',
                                'dataset_id': dataset_id,
                                'method_name': method_name,
                                'key_0': str(new_key[0]),
                                'key_1': str(new_key[1]),
                                'extra_suffix': '_n{}seed{}'.format(num_dataset, seed),
                            }
                        )
                        work_units.append(await experiment.add(job))

        for num_dataset in VARY_NUM_DATASET_NUM_LIST:
            for seed in range(VARY_NUM_DATASET_NUM_SEEDS):
                for dataset_id in FULL_ID_LIST:
                    for method_name in VARY_NUM_DATASET_METHOD_NAME_LIST:
                        if method_name == 'random':
                            continue
                        new_key, key = jax.random.split(key)
                        job = xm.Job(
                            executable=executable,
                            executor=xm_local.Vertex(requirements=requirements),
                            args={
                                'group_id': GROUP_ID,
                                'mode': 'eval_nll_setup_b_test_id',
                                'dataset_id': dataset_id,
                                'method_name': method_name,
                                'key_0': str(new_key[0]),
                                'key_1': str(new_key[1]),
                                'extra_suffix': '_n{}seed{}'.format(num_dataset, seed),
                            }
                        )
                        work_units.append(await experiment.add(job))

        for work_unit in work_units:
            await work_unit.wait_until_complete()

        print_and_say('merge results')

        spec = xm.PythonContainer(
            path='/home/zfan/hyperbo',
            base_image='python:3.7.11',
            entrypoint=xm.CommandList([
                'python3 hyperbo/experiments_xmanager/hpl_bo_split_worker.py $@',
            ]),
            docker_instructions=[
                'WORKDIR bo',
                'COPY hyperbo/hyperbo hyperbo',
                'COPY hyperbo/hpob hpob',
                'COPY hyperbo/setup.py setup.py',
                'RUN python3 -m pip install --upgrade pip==23.2',
                'RUN pip3 install .',
            ],
        )

        [executable] = experiment.package([
            xm.Packageable(
                executable_spec=spec,
                executor_spec=xm_local.Vertex.Spec(),
            ),
        ])

        requirements = xm.JobRequirements(
            cpu=BO_NODE_CPU_COUNT,
            ram=3.9 * BO_NODE_CPU_COUNT * xm.GiB,
        )

        new_key, key = jax.random.split(key)
        job = xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            args={
                'group_id': GROUP_ID,
                'mode': 'merge_vary_num_dataset',
                'dataset_id': '',
                'key_0': str(new_key[0]),
                'key_1': str(new_key[1])
            }
        )
        work_unit = await experiment.add(job)

        await work_unit.wait_until_complete()


if __name__ == '__main__':
    try:
        app.run(main)
    except Exception:
        print_and_say('The experiment has failed.')
        raise
