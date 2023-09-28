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
        time_0 = time.time()
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

        time_1 = time.time()

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

        time_2 = time.time()

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

        new_key, key = jax.random.split(key)
        job = xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            args={
                'group_id': GROUP_ID,
                'mode': 'fit_direct_hgp_two_step_setup_b',
                'key_0': str(new_key[0]),
                'key_1': str(new_key[1]),
            }
        )
        work_units.append(await experiment.add(job))

        for train_id in FULL_ID_LIST:
            new_key, key = jax.random.split(key)
            job = xm.Job(
                executable=executable,
                executor=xm_local.Vertex(requirements=requirements),
                args={
                    'group_id': GROUP_ID,
                    'mode': 'fit_direct_hgp_two_step_setup_b_leaveout',
                    'dataset_id': train_id,
                    'key_0': str(new_key[0]),
                    'key_1': str(new_key[1]),
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
            }
        )
        work_units.append(await experiment.add(job))

        for train_id in FULL_ID_LIST:
            new_key, key = jax.random.split(key)
            job = xm.Job(
                executable=executable,
                executor=xm_local.Vertex(requirements=requirements),
                args={
                    'group_id': GROUP_ID,
                    'mode': 'fit_hpl_hgp_two_step_setup_b_leaveout',
                    'dataset_id': train_id,
                    'key_0': str(new_key[0]),
                    'key_1': str(new_key[1]),
                }
            )
            work_units.append(await experiment.add(job))

        for work_unit in work_units:
            await work_unit.wait_until_complete()

        time_3 = time.time()

        print_and_say('fit end-to-end HGP')

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
            cpu=FITTING_E2E_NODE_CPU_COUNT,
            ram=15 * FITTING_E2E_NODE_CPU_COUNT * xm.GiB,
        )

        work_units = []

        new_key, key = jax.random.split(key)
        job = xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            args={
                'group_id': GROUP_ID,
                'mode': 'fit_hpl_hgp_end_to_end_setup_b',
                'key_0': str(new_key[0]),
                'key_1': str(new_key[1]),
            }
        )
        work_units.append(await experiment.add(job))

        for train_id in FULL_ID_LIST:
            new_key, key = jax.random.split(key)
            job = xm.Job(
                executable=executable,
                executor=xm_local.Vertex(requirements=requirements),
                args={
                    'group_id': GROUP_ID,
                    'mode': 'fit_hpl_hgp_end_to_end_setup_b_leaveout',
                    'dataset_id': train_id,
                    'key_0': str(new_key[0]),
                    'key_1': str(new_key[1]),
                }
            )
            work_units.append(await experiment.add(job))

        new_key, key = jax.random.split(key)
        job = xm.Job(
            executable=executable,
            executor=xm_local.Vertex(requirements=requirements),
            args={
                'group_id': GROUP_ID,
                'mode': 'fit_hpl_hgp_end_to_end_setup_b_no_init',
                'key_0': str(new_key[0]),
                'key_1': str(new_key[1]),
            }
        )
        work_units.append(await experiment.add(job))

        for train_id in FULL_ID_LIST:
            new_key, key = jax.random.split(key)
            job = xm.Job(
                executable=executable,
                executor=xm_local.Vertex(requirements=requirements),
                args={
                    'group_id': GROUP_ID,
                    'mode': 'fit_hpl_hgp_end_to_end_setup_b_leaveout_no_init',
                    'dataset_id': train_id,
                    'key_0': str(new_key[0]),
                    'key_1': str(new_key[1]),
                }
            )
            work_units.append(await experiment.add(job))

        for work_unit in work_units:
            await work_unit.wait_until_complete()

        time_4 = time.time()

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
            ram=8.0 * BO_NODE_CPU_COUNT * xm.GiB,
        )

        work_units = []

        for ac_func_type in AC_FUNC_TYPE_LIST:
            for test_id in FULL_ID_LIST:
                for method_name in ['hpl_hgp_end_to_end_from_scratch', 'hpl_hgp_end_to_end_leaveout_from_scratch']:
                # for method_name in METHOD_NAME_LIST:
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
                        }
                    )
                    work_units.append(await experiment.add(job))

        if IS_HPOB:
            for ac_func_type in AC_FUNC_TYPE_LIST:
                for method_name in PD1_METHOD_NAME_LIST:
                    new_key, key = jax.random.split(key)
                    job = xm.Job(
                        executable=executable,
                        executor=xm_local.Vertex(requirements=requirements),
                        args={
                            'group_id': GROUP_ID,
                            'mode': 'test_bo_setup_b_id',
                            'dataset_id': 'pd1',
                            'ac_func_type': ac_func_type,
                            'method_name': method_name,
                            'key_0': str(new_key[0]),
                            'key_1': str(new_key[1]),
                        }
                    )
                    work_units.append(await experiment.add(job))

        for work_unit in work_units:
            await work_unit.wait_until_complete()

        time_5 = time.time()

        print_and_say('evaluate training loss and NLL')

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

        if not IS_HPOB:
            for method_name in METHOD_NAME_LIST:
                if method_name not in EVAL_KL_METHOD_NAME_LIST:
                    continue
                job = xm.Job(
                    executable=executable,
                    executor=xm_local.Vertex(requirements=requirements),
                    args={
                        'group_id': GROUP_ID,
                        'mode': 'eval_kl_with_gt',
                        'method_name': method_name,
                    }
                )
                work_units.append(await experiment.add(job))

        key = jax.random.PRNGKey(0)
        for method_name in METHOD_NAME_LIST:
            if (not method_name.startswith('hpl_hgp')) or ('leaveout' in method_name):
                continue
            job = xm.Job(
                executable=executable,
                executor=xm_local.Vertex(requirements=requirements),
                args={
                    'group_id': GROUP_ID,
                    'mode': 'eval_loss_setup_b_train',
                    'method_name': method_name,
                    'key_0': str(key[0]),
                    'key_1': str(key[1]),
                }
            )
            work_units.append(await experiment.add(job))

            job = xm.Job(
                executable=executable,
                executor=xm_local.Vertex(requirements=requirements),
                args={
                    'group_id': GROUP_ID,
                    'mode': 'eval_loss_setup_b_test',
                    'method_name': method_name,
                    'key_0': str(key[0]),
                    'key_1': str(key[1]),
                }
            )
            work_units.append(await experiment.add(job))

        key = jax.random.PRNGKey(0)
        for i in range(EVAL_NLL_GP_RETRAIN_N_BATCHES):
            new_key, key = jax.random.split(key)
            for method_name in METHOD_NAME_LIST:
                if method_name == 'random':
                    continue
                if method_name not in ['hpl_hgp_two_step', 'hpl_hgp_end_to_end', 'hpl_hgp_end_to_end_from_scratch']:
                    continue
                for dataset_id in FULL_ID_LIST:
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
                            'eval_nll_gp_retrain_batch_id': i,
                        }
                    )
                    work_units.append(await experiment.add(job))

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
                            'eval_nll_gp_retrain_batch_id': i,
                        }
                    )
                    work_units.append(await experiment.add(job))

            if IS_HPOB:
                for method_name in PD1_METHOD_NAME_LIST:
                    if method_name == 'random':
                        continue
                    if method_name not in ['hpl_hgp_two_step', 'hpl_hgp_end_to_end', 'hpl_hgp_end_to_end_from_scratch']:
                        continue
                    job = xm.Job(
                        executable=executable,
                        executor=xm_local.Vertex(requirements=requirements),
                        args={
                            'group_id': GROUP_ID,
                            'mode': 'eval_nll_setup_b_train_id',
                            'dataset_id': 'pd1',
                            'method_name': method_name,
                            'key_0': str(new_key[0]),
                            'key_1': str(new_key[1]),
                            'eval_nll_gp_retrain_batch_id': i,
                        }
                    )
                    work_units.append(await experiment.add(job))

                    new_key, key = jax.random.split(key)
                    job = xm.Job(
                        executable=executable,
                        executor=xm_local.Vertex(requirements=requirements),
                        args={
                            'group_id': GROUP_ID,
                            'mode': 'eval_nll_setup_b_test_id',
                            'dataset_id': 'pd1',
                            'method_name': method_name,
                            'key_0': str(new_key[0]),
                            'key_1': str(new_key[1]),
                            'eval_nll_gp_retrain_batch_id': i,
                        }
                    )
                    work_units.append(await experiment.add(job))

        for work_unit in work_units:
            await work_unit.wait_until_complete()

        time_6 = time.time()

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
                'mode': 'merge',
                'dataset_id': '',
                'key_0': str(new_key[0]),
                'key_1': str(new_key[1])
            }
        )
        work_unit = await experiment.add(job)

        await work_unit.wait_until_complete()

        time_7 = time.time()

        print_and_say('total time: {}'.format(time_7 - time_0))
        print_and_say('config time: {}'.format(time_1 - time_0))
        print_and_say('fit single GP time: {}'.format(time_2 - time_1))
        print_and_say('fit two-step GP time: {}'.format(time_3 - time_2))
        print_and_say('fit end-to-end GP time: {}'.format(time_4 - time_3))
        print_and_say('run BO time: {}'.format(time_5 - time_4))
        print_and_say('evaluate NLL time: {}'.format(time_6 - time_5))
        print_and_say('merge results time: {}'.format(time_7 - time_6))


if __name__ == '__main__':
    try:
        app.run(main)
    except Exception:
        print_and_say('The experiment has failed.')
        raise
