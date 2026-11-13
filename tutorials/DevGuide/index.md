# 05 How to Add a New Architecture[¶](#05-how-to-add-a-new-architecture)

If you want to add new autoencoder architectures and pipelines, this is highly encouraged!

Let’s say you find a publication with an interesting architecture called *MySpecialix*.\
Here we show how to incorporate this into the *AUTOENCODIX* framework, so that you can use this new architecture in the same way you would use an existing pipeline such as *Varix*.

```
result = varix.run(config=config)
# then you could also do
special_result = myspecialix.run(config=config)
```

## 1) High-Level Overview[¶](#1-high-level-overview)

### 1.1) High-Level Workflow[¶](#11-high-level-workflow)

To add a new autoencoder architecture, you need to implement at least two components:

- The pipeline in *src/autoencodix/* — analogous to *varix.py*.
- One additional piece of functionality, most likely a new architecture, but this may also be:
  - a new loss function for an existing architecture, or
  - a new method inside the pipeline.

Depending on complexity and data requirements, you may also add:

- The model architecture in *src/autoencodix/modeling/* — analogous to *\_varix_architecture.py*.
- A custom dataset class in *src/autoencodix/data* — analogous to *numeric_dataset.py*.
- A custom preprocessor in *src/autoencodix/data* — as in *preprocessor.py*.
- A custom trainer in *src/autoencodix/trainers* — as in *\_general_trainer.py*, including a custom *predict* method if needed.
- A custom loss — as in *xmodal_loss.py*.
- A custom visualizer — as in *\_xmodal_visualizer.py*.
- A custom evaluator for downstream tasks — as in *\_xmodalix_evaluator.py*.

### 1.2) High-Level Structure[¶](#12-high-level-structure)

- Every Autoencodix model is built on the base classes in *src/autoencodix/base*. These (often abstract) classes define the structure for each pipeline stage: preprocessing, fitting, predicting, evaluating, and visualizing. Additional components such as losses are also defined here.
- The base classes provide shared functionality, such as how trainers or preprocessors are invoked.
- The base classes should not be modified. If custom behavior is needed, override the relevant methods in your child class.

## 2) Implementation Details[¶](#2-implementation-details)

### 2.1) Required Files — Detailed Steps[¶](#21-required-files-detailed-steps)

Example: You want to add a new architecture named **MySpecial**.

#### Add the architecture file[¶](#add-the-architecture-file)

Create the file:

*src/autoencodix/modeling/\_myspecialix_architecture.py*

(Files not intended for end-user import should start with an underscore.)

Define your class, for example *MySpecialixArchitecture*.

#### Inherit from the base architecture[¶](#inherit-from-the-base-architecture)

Your class should inherit from the base architecture in:

*src/autoencodix/base/\_base_architecture.py*

#### Review the base class[¶](#review-the-base-class)

Understand which abstract methods you must implement.

> Hint: Check existing implementations, such as:

*src/autoencodix/modeling/\_varix_architecture.py*

#### Add new configuration parameters[¶](#add-new-configuration-parameters)

If your architecture needs new parameters, add them to:

*src/autoencodix/configs/default_config.py*

You may later create a custom config for your pipeline, but adding new parameters to the *DefaultConfig* first — with sensible defaults — is recommended.

#### Expose your class in ***init**.py*[¶](#expose-your-class-in-initpy)

Add *MySpecialixArchitecture* to:

*src/autoencodix/modeling/**init**.py*

Note:

> Do this for all new classes you add (trainers, preprocessors, losses, etc.). .Always check the base classes first; many functionalities can be reused.

#### Write tests[¶](#write-tests)

Create a test file:

*tests/test_modeling/test_myspecialix_architecture.py*

## Additional Explanation: Passing Types Instead of Instances[¶](#additional-explanation-passing-types-instead-of-instances)

Most pipeline functionality comes from the *BasePipeline*.\
To customize behavior for the *MySpecialix* pipeline, we pass types of specialized subclasses rather than instances.

Reason:

- During *BasePipeline* initialization, not all parameters required for constructing your specialized subclasses are available yet.
- Therefore, only the class types are passed.
- *BasePipeline* later creates the actual instances once all required parameters are known.

All passed types must inherit from the corresponding base classes.

## Optional Files[¶](#optional-files)

Optional components follow the same pattern as the required ones:

- Create a specialized class inheriting from the appropriate base class.
- Pass the type of this specialized class (for example, *MySpecialTrainer*) to your *MySpecialix* pipeline in its ***init*** method — just like you did with *MySpecialixArchitecture*.

This design keeps pipelines modular, flexible, and consistent across the Autoencodix package.
