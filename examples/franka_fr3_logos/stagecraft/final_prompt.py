import textwrap

prompt_template = textwrap.dedent("""
You are a robot learning policy execution expert who has to gauge the performance of a policy that will be executed in a particular environment with a FRANKA EMIKA Research 3 robot:
            
            You are provided with {n_videos} inital environment state images. Each image consists of "front_camera_RGB, wrist_camera_RGB" frames showing the progression of a policy rollout.
            The images are labeled v1 through v{n_videos}. Additionally, you are provided with the policy rollout results that are either "SUCCESS" or "FAILURE" tags. The image labels and the result tags are as follows:
            
            {video_labels}
            
            The policy was instructed with the following task instruction: {task_instruction} 
            In order to gauge the performance of the policy rollouts, you need to understand the following prerequisites:
                    1. [object] : This is a unique object present in the robot's workspace.
                    2. < object_1, object_2, ... > : This is an environment category that consists of a unique set of [object]s present in the robot's workspace.
                            
                        For example: (i) If in the robot's workspace a [coffee_cup], and a [water_bottle] are visible then the environment category is given as < coffee_cup, water_bottle >
                                    (ii) If in the robot's workspace a [red_cube] is visible then the environment category is given as <red_cube>
                        
                    3. @action(object) :  This is a predefined primitive action that the robot can take to change the environment category of the workspace.
                                          List of primitives available with the robot are: [@remove]
                                             
                        For example: (i) The primitive action @remove(coffee_cup) will change the environment category from < coffee_cup, water_bottle > to < water_bottle >
                                     (ii) The primitive action @add(coffee_cup) will change the environment category from < coffee_cup, water_bottle > to < coffee_cup, water_bottle, coffee_cup >

            Now you have to follow the algorithm below that describes your job as a policy execution expert:
                                             
            ****ALGORITHM****
            1. Identify the [object]s present in the robot's workspace across all the {n_videos} inital environment state-image.
               Make sure you assign rich semantic descriptions for the [object]s. For example: [blue_water_bottle] and [green_water_bottle] are two different [object]s and should not be generalized as a single [water_bottle] object.
               Then build the corresponding environment category for each image. NOTE: Multiple images might have the same environment category if they have the same set of [object]s.
               
                                             
            2. Identify the unique environment categories and assign success rates to each of them based on the result tags associated with the inital environment state image.
                    For example: If the environment category < coffee_cup, water_bottle > appears 5 times in the inital environment state-image and the result tags for this environment category are
                                 SUCCESS, SUCCESS, FAILURE, SUCCESS, SUCCESS respectively, then the success rate for the environment category will be 4/5 or 0.8
                                             
            
            3. Identify the environment categories with high success rates. NOTE: These are our target environment categories.
         
            4. Consider the "new_initial_state_image" provided at the end of the initial environment state image sequence. This is the initial state image of a new robot workspace.
                
                Identify the environment category for this "new_initial_state_image" using step 1 and step 2 of the ALGORITHM. NOTE: The environment category for this workspace may be different from all the
                environment categories previously identified from the {n_videos} initial environment state images, and no performance tag will be available for the "new_initial_state_image".
                
            5. Now your task is to deduce a sequence of primitive @action to change the environment category of the "new_initial_state_image" into an environment category that has a high success rate.
                NOTE: Only remove the necessary objects to reach the desired environment category with a high success rate based on the observed empirical performance. 
                You are only allowed to use primitive @action from the list of available actions as mentioned in the prerequisites.
                                                                              
                                             
            Your output must be in the following format:

            {{
                "policy_execution_analysis": {{
                    "object_list": [
                        "[object_1]", "[object_2]", "[object_3]", ...
                    ],
                    "unique_environment_categories": [
                        "< object_1, object_4, ... >",
                        "< object_4, object_6, ... >",
                        "< object_9, object_12, ... >",
                        ...
                    ],
                    "success_rates": [
                        "SR1%", "SR2%", "SR3%", ...
                    ],
                    "new_initial_state_image_environment_category": "< object_5, object_4, ... >",
                    "deduced_actions": [
                        "@action1(object1)", "@action2(object2)", "@action3(object3)", ...
                    ],
                    "rationale": "Your reasoning..."
                }}
            }}

            Field descriptions:
            - "object_list": List of objects present in the workspace across all the inital environment state-images and the "new_initial_state_image".
            - "unique_environment_categories": The list of unique environment categories across all the inital environment state-images.
            - "success_rates": Success rates of all the unique environment categories. The cardinality of this list should be the same as that of unique_environment_categories.
            - "new_initial_state_image_environment_category": The environment category of the "new_initial_state_image".
            - "deduced_actions": Sequence of deduced primitive actions that results in a high success rate environment category.
            - "rationale": A thorough yet succinct explanation of your reasoning for the sequence of deduced actions.

            NOTE: Please ensure all the values in the above JSON are strings or lists of strings.

            Respond ONLY with a valid JSON object matching the structure above. Do not include any prose, explanation, or markdown formatting outside the JSON block.
            """)

