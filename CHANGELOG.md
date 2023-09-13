# CHANGELOG



## v0.10.0 (2023-09-13)

### Chore

* chore: updating semantic release config ([`b30ea37`](https://github.com/chanind/frame-semantic-transformer/commit/b30ea37644ad5c400bad46ec1b22039144559335))

### Feature

* feat: remove various transformer warnings and fix training documentation (#25)

Co-authored-by: Curtis Ruck &lt;ruckc@DESKTOP-ME5SH6R&gt; ([`6f2e1d1`](https://github.com/chanind/frame-semantic-transformer/commit/6f2e1d11fc8348cd67cf6d6b512a43b4a8a6c00b))

### Fix

* fix: default sampling to False to avoid changing existing behavior ([`46260d6`](https://github.com/chanind/frame-semantic-transformer/commit/46260d64e0b83350a0d5da594080b59bd9587cf5))

* fix: linting ([`b331dfa`](https://github.com/chanind/frame-semantic-transformer/commit/b331dfa42233315a510abf031f755c736a3dd022))


## v0.9.0 (2023-06-08)

### Chore

* chore: adding torch 1.13 to dev deps to help CI run tests ([`8abff7b`](https://github.com/chanind/frame-semantic-transformer/commit/8abff7b62d962cbbdf9c27ed3bfa849b4b0cb7f2))

* chore: updating docs to discourage multiple sentences per string ([`e77c5a2`](https://github.com/chanind/frame-semantic-transformer/commit/e77c5a28819d3f6ab7a4bf24289e83559801ed05))

* chore: adding an integration test for unk chars strings ([`1c21345`](https://github.com/chanind/frame-semantic-transformer/commit/1c213450aeec1b4aeea500627db683ffbe0c6e26))

### Feature

* feat: removing logger config for release ([`7947f8b`](https://github.com/chanind/frame-semantic-transformer/commit/7947f8bac4e4e2f6e9c1d437dbdcccbde0880132))

### Unknown

* Add ability to set which gpu to use (#23)

* updated handling of &#39;use_gpu&#39; option to allow specifying gpu index to use

* added some error handling with logging messages to the FrameSemanticTransformer constructor

* fixed the last update so that the format string now is only set for local logger and not for the root logger ([`c62aa07`](https://github.com/chanind/frame-semantic-transformer/commit/c62aa079ed6ee2698ac4b0aef59c7f11a8f821e9))


## v0.8.2 (2023-04-15)

### Chore

* chore: updating bibtex ([`0e4af39`](https://github.com/chanind/frame-semantic-transformer/commit/0e4af39678898c8ec0bf7003668ffa4a30e7b8b6))

* chore: pin poetry v1.4.0 so furo will install in CI ([`24dc8c7`](https://github.com/chanind/frame-semantic-transformer/commit/24dc8c7583bcf37d6e1d79f2bcab21cb811c322b))

* chore: pin poetry v1.4.0 so furo will install in CI ([`f8ca051`](https://github.com/chanind/frame-semantic-transformer/commit/f8ca051faf47a8a385d7971a6ad27360e0d6347a))

* chore: add CITATION.cff ([`223cfce`](https://github.com/chanind/frame-semantic-transformer/commit/223cfce5ebe80ac3fb303481768ad2d4e0300bd3))

* chore: adding citation bibtex info to README ([`b5435fa`](https://github.com/chanind/frame-semantic-transformer/commit/b5435fa7d8d60314c37d82bd0aeded80c785b105))

### Fix

* fix: Align trigger-marked sentence to original sentence (#19)

* trigger identification task alignment fix

* black formatting

* updated test cases

---------

Co-authored-by: Jacob Striebel &lt;striebel@users.noreply.github.com&gt; ([`6683a22`](https://github.com/chanind/frame-semantic-transformer/commit/6683a224740ddd8a19a707247f732bfc9e694ba2))


## v0.8.1 (2023-03-15)

### Fix

* fix: auto-download omw-1.4 for inference ([`343906c`](https://github.com/chanind/frame-semantic-transformer/commit/343906c3a0df59cd111355d8aa84d33b39003459))


## v0.8.0 (2023-03-15)

### Feature

* feat: new models trained on Framenet exemplars (#18)

* include exemplars in framenet training

* skipping invalid trigger exemplars

* skip exemplars by default during training

* fixing tests

* improving data augmentations

* ensure wordnet download for inference

* updating snapshots

* adding more info when augmentations fail validation

* adding more augmentations from nlpaug

* fixing linting

* fixing keyboard augmentation

* more checks on keyboard augmentation

* tweaking augmentations

* fixing tests

* adding safety check to uppercase augmentation

* lower augmentation rate

* adding more augmentations

* tweaking augs

* removing debugging output

* reduce augmentation

* tweaking augmentation probs

* tweaking augmentation probs

* fixing type import

* adding option to delete non-optimal models as training progresses

* tweaking augmentations

* updating models

* updating README with new model stats ([`3f937fb`](https://github.com/chanind/frame-semantic-transformer/commit/3f937fbc475bd8600cb62ca885fcf0e4a80effba))


## v0.7.0 (2023-03-09)

### Chore

* chore: serialize eval logs before writing json ([`40983ff`](https://github.com/chanind/frame-semantic-transformer/commit/40983ff269e81b5d8ecab2ea1f0a9935a73d08dc))

* chore: fixing missing links for readthedocs ([`fbca04e`](https://github.com/chanind/frame-semantic-transformer/commit/fbca04e90c600a0a4a7dfa166e0a3d850b5f7b65))

* chore: explicitly installing furo in readthedocs ([`a581825`](https://github.com/chanind/frame-semantic-transformer/commit/a58182530fc13891b0ec153c0f613727d1078ebc))

* chore: try using python 3.9 for readthedocs build ([`55c712f`](https://github.com/chanind/frame-semantic-transformer/commit/55c712f53f991716571df43c29461819f3903a22))

* chore: manually install torch in readthedocs ([`1ad58a9`](https://github.com/chanind/frame-semantic-transformer/commit/1ad58a92220f62d70015dea29dfd131a3d5392d6))

* chore: setting up docs page with sphinx and readthedocs (#17)

* setting up docs page with sphinx and readthedocs

* ignore docs for flake8 linting ([`8d3c5fd`](https://github.com/chanind/frame-semantic-transformer/commit/8d3c5fd1ac595af3b01f143fd8ce960dae883e1c))

* chore: create outputs dir when logging if it doesn&#39;t exist ([`7629560`](https://github.com/chanind/frame-semantic-transformer/commit/7629560a8d3657674c7c4d8500e464ada0faa203))

* chore: adding option to log eval failures ([`f68fb61`](https://github.com/chanind/frame-semantic-transformer/commit/f68fb618d14d45f132f387bf4a12d9d29ca13b8c))

### Feature

* feat: Propbank (#16)

* setting up propbank loaders

* skip light verbs for now

* trying a different approach to avoid lv verbs

* fixing typo

* just ignore non-existent frames

* use similar lu norm to framenet

* adding option to resume from checkpoint

* switching to propbank 3.4 instead of 3.1

* fixing propbank nltk paths

* removing debuggin prints

* fixing test

* adding optional LR decay ([`4c53887`](https://github.com/chanind/frame-semantic-transformer/commit/4c538875be45c5b851286b1a17156b951c786a07))

### Unknown

* add readthedocs link to readme ([`bbeeed8`](https://github.com/chanind/frame-semantic-transformer/commit/bbeeed8ac9d7d348514b1067fb2ccbca8466a8e8))


## v0.6.2 (2023-03-01)

### Performance

* perf: minor performance improvements for arg extraction and frame detection (#15)

* save best model chkpt based on val loss

* Adding loader setup to training

* add more optional config for loggers/callbacks during training

* adding explicit logging for test/train/val loss to end of epochs

* rever to default PL logging behavior if no loggers are provided

* adding helpers for model evaluations

* try to standardize arg extraction output

* standardize punct in args extraction

* use fast tokenizer for sent clenanup

* switch to just using tokenizer cleanup for speed

* run clean_up_tokenization just once before arg extraction, not for each arg

* fixing val_metrics err ([`7e03969`](https://github.com/chanind/frame-semantic-transformer/commit/7e039695b8cfa1ae5b8de1b9b4d4145e5bec3884))


## v0.6.1 (2023-02-27)

### Chore

* chore: remove poetry.lock from demo docker build ([`2f8ffb9`](https://github.com/chanind/frame-semantic-transformer/commit/2f8ffb99afa82f259db6043247d8b227a829a44a))

### Fix

* fix: fixing errors when no frames are found (#14) ([`ef2424c`](https://github.com/chanind/frame-semantic-transformer/commit/ef2424cba57515b9aa8205fe29f015b57f26504e))


## v0.6.0 (2023-02-23)

### Feature

* feat: adding support for running inference on multiple sentences in batches (#11) ([`e6423e5`](https://github.com/chanind/frame-semantic-transformer/commit/e6423e594de22850d9ea6369f80ca9a0e81ac8e0))


## v0.5.0 (2023-02-22)

### Chore

* chore: fixing README badge after shields.io breaking change ([`bd90fef`](https://github.com/chanind/frame-semantic-transformer/commit/bd90fef60aa1ae96c672685f0315389ac6839651))

### Feature

* feat: Multilingual training refactor (#10)

* WIP refactoring to make it easier to train on different framenets

* Making evaluate runnable directly to evaluate pretrained models

* tweaking tests

* refactoring training / eval scripts

* add validation that loaders match model

* updating README

* cleaning up typing

* use 3.8 for CI

* updating semantic release ([`7bf7ae5`](https://github.com/chanind/frame-semantic-transformer/commit/7bf7ae5474cc586739f1ad77848e45db09ff5a0d))

### Unknown

* fix typo in README (#6) ([`a151e5a`](https://github.com/chanind/frame-semantic-transformer/commit/a151e5aaf87ef5ece2add1a68968cbde068a5d3c))

* Update README.md ([`461eced`](https://github.com/chanind/frame-semantic-transformer/commit/461eced8ea745b63154cff0c44a9d16d8be80812))

* Update README.md ([`a6f5001`](https://github.com/chanind/frame-semantic-transformer/commit/a6f500151baa241105fee05f22383b86d923d9f6))


## v0.4.1 (2022-05-25)

### Fix

* fix: updating README stats ([`76e4e75`](https://github.com/chanind/frame-semantic-transformer/commit/76e4e754b534389de9c86caced762461ae884e88))


## v0.4.0 (2022-05-24)

### Feature

* feat: Frame classification hints (#3)

* adding in lexical unit data for smarter frame classification

* adding in stemming for lu handling

* allow skipping validation in initial epochs for faster training

* use self.current_epoch instead of batch_idx

* using bigrams to reduce the amount of frame suggestions

* refactoring bigrams stuff and adding more tests

* fixing bug with trigger bigrams

* updating README

* updating model revision ([`201ed51`](https://github.com/chanind/frame-semantic-transformer/commit/201ed517c7a78b32d659150dadf11fcd7b4fea6d))

### Unknown

* fixing typo in demo server ([`a15ef6d`](https://github.com/chanind/frame-semantic-transformer/commit/a15ef6da75483e016cb8959441ccf90e5ec79700))

* improving demo UI (#4)

* improving demo UI

* adding secret &#39;model&#39; param to client ([`8bf3275`](https://github.com/chanind/frame-semantic-transformer/commit/8bf327592f406b821bee2dca9fb58d85b0b200ab))

* UI improvements for demo ([`69e85af`](https://github.com/chanind/frame-semantic-transformer/commit/69e85afc30e4835f2fdd5bc7f5cd6dbb7a8b021a))


## v0.3.3 (2022-05-22)

### Fix

* fix: make trimmed batch contiguous (#2) ([`21aee70`](https://github.com/chanind/frame-semantic-transformer/commit/21aee707b25a96139f19c84c176fe692323e6b48))


## v0.3.2 (2022-05-22)

### Fix

* fix: add torch.no_grad() to batch trimming step ([`8b8a401`](https://github.com/chanind/frame-semantic-transformer/commit/8b8a401b339b674abeca300408d613472f6fa197))


## v0.3.1 (2022-05-22)

### Fix

* fix: adding LICENSE into pypi description ([`c6e0a42`](https://github.com/chanind/frame-semantic-transformer/commit/c6e0a42fc606cc06399f67966338ebbd8bc0e5d9))

* fix: adding README into pypi description ([`1b99551`](https://github.com/chanind/frame-semantic-transformer/commit/1b9955136ff6f4679aa093fd7a05765c035f03f4))


## v0.3.0 (2022-05-22)

### Chore

* chore: adding badges to README ([`ac00793`](https://github.com/chanind/frame-semantic-transformer/commit/ac007933abb0ca3f54aa9bda0a1d703a5873b3f7))

### Feature

* feat: adding a helper to trim unnecessary padding chars for faster training / generation (#1) ([`58e58a8`](https://github.com/chanind/frame-semantic-transformer/commit/58e58a8ee19da5a57b24d6aa8ab0a440fc9e1c93))


## v0.2.1 (2022-05-22)

### Fix

* fix: reverting to older lock file for mypy ([`08f0c63`](https://github.com/chanind/frame-semantic-transformer/commit/08f0c638f02d7b2dfd0589b8ace2232dfd3e9fb8))

* fix: relaxing transformers version req ([`57464a1`](https://github.com/chanind/frame-semantic-transformer/commit/57464a19cb69a21f91a1477ec44cc5ee09a29be2))


## v0.2.0 (2022-05-21)

### Feature

* feat: autopublish ([`a0900ff`](https://github.com/chanind/frame-semantic-transformer/commit/a0900ff546e99be008491b9b2446d9534e08af76))

### Fix

* fix: pinning old version of semantic-release plugin ([`85f3a62`](https://github.com/chanind/frame-semantic-transformer/commit/85f3a625225da5c880b929c147bcf65bd629f2ea))

* fix: adding fetch-depth 0 for release ([`6dab4e6`](https://github.com/chanind/frame-semantic-transformer/commit/6dab4e661059cc8e895161cc07e3bec84719f784))

* fix: autopublish to pypi ([`3591c27`](https://github.com/chanind/frame-semantic-transformer/commit/3591c27a78ff7a687d378649b806dfacce00d775))

### Unknown

* restrict model revisions ([`9a36fc4`](https://github.com/chanind/frame-semantic-transformer/commit/9a36fc419798422140826caefb21a1b355fa6b8c))

* adding explanation about lightning_logs dir ([`7898bba`](https://github.com/chanind/frame-semantic-transformer/commit/7898bbab8a6cbcbbb96c4d329bc4be64afcd1ae3))

* updating README and improving train script ([`94d7fac`](https://github.com/chanind/frame-semantic-transformer/commit/94d7fac13a3eae417cc6410f9ca973e568a5210a))

* Create LICENSE ([`f73fc0e`](https://github.com/chanind/frame-semantic-transformer/commit/f73fc0ea4720d960d499e92fb5fee07ed6f68579))

* augment training samples dynamically during training ([`3b40f07`](https://github.com/chanind/frame-semantic-transformer/commit/3b40f07f3538a7551a808220f24e7b89e5366bf5))

* adding tests for chain_augmentations ([`57f24dc`](https://github.com/chanind/frame-semantic-transformer/commit/57f24dc3ee559718dcb255e9ce2c752dd6522962))

* adding write permissions to publish job ([`d4b3e22`](https://github.com/chanind/frame-semantic-transformer/commit/d4b3e2264cc5e7b82d7a45c426b95beb00d854e3))

* try checkout v2 ([`7883323`](https://github.com/chanind/frame-semantic-transformer/commit/78833239d4735aaf52226c0d8d22be9d84dd229f))

* try adding token to checkout action ([`dce1651`](https://github.com/chanind/frame-semantic-transformer/commit/dce1651cdaf32b028f1c5e491e069c141da894a3))

* adding link to demo in README ([`f0b632b`](https://github.com/chanind/frame-semantic-transformer/commit/f0b632b09c921083910d93b8e2d3a2fd9ae3f9d6))

* fix node version in gh action ([`7db37cb`](https://github.com/chanind/frame-semantic-transformer/commit/7db37cb1f061ef460931a124127fefe3f864fbcd))

* add an action to publish the website ([`182504f`](https://github.com/chanind/frame-semantic-transformer/commit/182504f3dfb5049f6ed45030783dc5da5442fdd5))

* augment data for train but not val or test ([`660919e`](https://github.com/chanind/frame-semantic-transformer/commit/660919eba8a1637871bd712f1c3a7b033412ce40))

* adding data augmentation ([`13de3f4`](https://github.com/chanind/frame-semantic-transformer/commit/13de3f46d37deafe8bca94eb545b993d7ee71865))

* adding small size model and lazy-loading for the nltk + models ([`bda0ca8`](https://github.com/chanind/frame-semantic-transformer/commit/bda0ca81e9ed71d37ee3f508043e0d3d53bc1fc7))

* adding a demo client using create-react-app ([`2a673d1`](https://github.com/chanind/frame-semantic-transformer/commit/2a673d10f49ef4b444416489b723b679fb7c31af))

* try restricting batch size to 2 to avoid excessive memory use ([`4f2e58c`](https://github.com/chanind/frame-semantic-transformer/commit/4f2e58caffca5cd27b26b7b0d4b658afde62f9dd))

* try reducing to 1 thread to save memory ([`cfc4d21`](https://github.com/chanind/frame-semantic-transformer/commit/cfc4d216f13f2b78b037ad68f075a428253ad238))

* adding cors support to flask ([`4d463cb`](https://github.com/chanind/frame-semantic-transformer/commit/4d463cb453788b5149f7f498ff796d4c670045fa))

* increase gunicorn timeout ([`acd42a1`](https://github.com/chanind/frame-semantic-transformer/commit/acd42a14b89ad6aa34ee35affab1c2eb123ec83f))

* try adding poetry.lock to speed up docker build ([`86c9bcb`](https://github.com/chanind/frame-semantic-transformer/commit/86c9bcbb911ae6d9784806fa704e4f44e40bd401))

* bump to trigger cloud run build ([`c56e5b4`](https://github.com/chanind/frame-semantic-transformer/commit/c56e5b4145a8b7d8a2b43bb58aa3b3d668329d7c))

* bump to trigger cloud run build ([`a6d3094`](https://github.com/chanind/frame-semantic-transformer/commit/a6d30943222a3115baa3fa7ba98cce68b12aa2e2))

* remove poetry.lock from docker build ([`805adf4`](https://github.com/chanind/frame-semantic-transformer/commit/805adf4f5fee3e23219fd0f62684b4f8972ad6c5))

* adding a dockerizer flask server for demo purposes ([`8f192ac`](https://github.com/chanind/frame-semantic-transformer/commit/8f192acbf1058b582e343610b0f4f0c1eaf6d831))

* fixing typo ([`b7b9e34`](https://github.com/chanind/frame-semantic-transformer/commit/b7b9e34c232e6760f582aecf18a8b9b08a6cb8f8))

* adding a base FrameSemanticTransformer class to make it easy to parse sentences into frames ([`c0b78cf`](https://github.com/chanind/frame-semantic-transformer/commit/c0b78cfd5356e990302550f81dcbf4f75250727e))

* refactoring TaskSample classes into Tasks and TaskSamples ([`667f85e`](https://github.com/chanind/frame-semantic-transformer/commit/667f85ec8314e5d9922612904d99125fc568da51))

* fixing tests ([`eab2f96`](https://github.com/chanind/frame-semantic-transformer/commit/eab2f966268f08c95d951b824c8f60378c824b8e))

* more efficient loading of frame elements from framenet ([`4655768`](https://github.com/chanind/frame-semantic-transformer/commit/465576894ad3a7fda21eeca65c84fa3b7deabef5))

* add a check for invalid output in eval ([`0e9eb88`](https://github.com/chanind/frame-semantic-transformer/commit/0e9eb88ad70f103464acd47431dde1d5f52c0ae4))

* add a check for invalid output in eval ([`81a829a`](https://github.com/chanind/frame-semantic-transformer/commit/81a829a0593af001d9b4041e92532771909869d4))

* eval arg id similar to how sesame does it ([`73b2db9`](https://github.com/chanind/frame-semantic-transformer/commit/73b2db9bae939554d1a97f101b308eecbd707486))

* try adding in all possible frame elements into task intro for argument extraction ([`94e3c89`](https://github.com/chanind/frame-semantic-transformer/commit/94e3c89557fe0931ecc243e736d8d118e52318ac))

* updating frame id samples to be closer to how sesame does it ([`39376e5`](https://github.com/chanind/frame-semantic-transformer/commit/39376e548755cc32939c6097231f33cb854aad0d))

* fixing evaltuate function to work with batches predictions ([`7ca93d1`](https://github.com/chanind/frame-semantic-transformer/commit/7ca93d112eb35c61574dfb03f2e0052ef77a600f))

* force tranformers v4.18.0 to keep mypy happy ([`ac2c6a6`](https://github.com/chanind/frame-semantic-transformer/commit/ac2c6a65be536492099123c541b594bbf89e57f1))

* using multiple predictions when evaluating frame id task ([`d28961c`](https://github.com/chanind/frame-semantic-transformer/commit/d28961cd5f87d42e87d51219f293984d06ef1efc))

* fixing typo ([`197b93f`](https://github.com/chanind/frame-semantic-transformer/commit/197b93ffee1b38110ee0c3508966835c0e2d797d))

* trying to add eval into training ([`d22080b`](https://github.com/chanind/frame-semantic-transformer/commit/d22080bdc1801645aac4be6caad5e9062c37bbac))

* limiting task rebalancing ratio ([`451eb6c`](https://github.com/chanind/frame-semantic-transformer/commit/451eb6c0c7db51a0ad50d847f3270bd3283d13df))

* adding in task mix balancing ([`ecde0b2`](https://github.com/chanind/frame-semantic-transformer/commit/ecde0b2005c3f7a30698fa88528a3d41cd34e6cc))

* moving T5Tokenizer.cleanup into standardize_punct method ([`12ccf38`](https://github.com/chanind/frame-semantic-transformer/commit/12ccf38253566325718a226c776274175128a235))

* Trying out built-in clean up tokenization method ([`5f4a723`](https://github.com/chanind/frame-semantic-transformer/commit/5f4a723f187e6a21cba1201c71326089b426593c))

* allow tweaking predict params in eval ([`a549041`](https://github.com/chanind/frame-semantic-transformer/commit/a5490414a914ddc5b975b747ce4302eeacd83522))

* tweaking trigger processing to hopefully be more amenable to how the tokenizer works ([`2103f4d`](https://github.com/chanind/frame-semantic-transformer/commit/2103f4dd008b3e5e9856cbc19caac043f0e4578f))

* more readable eval print ([`726126d`](https://github.com/chanind/frame-semantic-transformer/commit/726126ddc977212a52963566de8fada7b48e188d))

* adding option to print eval failures ([`d23bf6f`](https://github.com/chanind/frame-semantic-transformer/commit/d23bf6f2fc2e7d0488f308ae1407d8f34ec862e0))

* adding a punct standardization step ([`d08877e`](https://github.com/chanind/frame-semantic-transformer/commit/d08877e0678c58abfe0d14e64c6673e68c03fb59))

* fixing linting ([`9250636`](https://github.com/chanind/frame-semantic-transformer/commit/9250636167d468ebd6df9aa6b20e8c9dbcfea495))

* tweaking frame id eval to match sesame logic ([`162f50f`](https://github.com/chanind/frame-semantic-transformer/commit/162f50fbcd26b20f0016312eab7cf0343c67393c))

* removing sample from dataloader, as it appears to break things ([`0e1dee0`](https://github.com/chanind/frame-semantic-transformer/commit/0e1dee00a9271b241a1f2d4446eaddc7e3cad407))

* fixing trigger samples and adding tests ([`6f19673`](https://github.com/chanind/frame-semantic-transformer/commit/6f1967352d01fc7305651ea57d73af689455d1f1))

* adding logging statements inside training function ([`a28a706`](https://github.com/chanind/frame-semantic-transformer/commit/a28a706238309f996bdb9ab3873d5bb4eb7f3fac))

* refactoring based on simple-t5 ([`d89e5db`](https://github.com/chanind/frame-semantic-transformer/commit/d89e5dbefd88db186ccbd0d1f70c42f65870226e))

* fixing evaluate typing ([`3cd3c33`](https://github.com/chanind/frame-semantic-transformer/commit/3cd3c33cf51434890783930f5a17470dd0d6ec52))

* fixing future annotations ([`16a227c`](https://github.com/chanind/frame-semantic-transformer/commit/16a227c846225c9f724de0ffc912118df5c61208))

* fixing bug in py 3.7 ([`ea953fc`](https://github.com/chanind/frame-semantic-transformer/commit/ea953fc1e4583b0ed7e41f7e1722cba1a4b3a9d1))

* refactoring and adding a target id task ([`37302bb`](https://github.com/chanind/frame-semantic-transformer/commit/37302bbfff7163df0199d8388e166c7b73f08bab))

* adding total to tqdm iteration ([`058a12c`](https://github.com/chanind/frame-semantic-transformer/commit/058a12c6db059322d2fee80c93a232a4ce5a26f1))

* fixing device issues ([`49a01f3`](https://github.com/chanind/frame-semantic-transformer/commit/49a01f32b9298c422289cc663f31e76aca5a327d))

* fixing typing ([`6012b09`](https://github.com/chanind/frame-semantic-transformer/commit/6012b092c355a4e79c74c474c30e884f2cd9d265))

* more efficient eval processing ([`4bbee5c`](https://github.com/chanind/frame-semantic-transformer/commit/4bbee5c813a5bb6daa3db7aca3c62b87637b1afd))

* add tqdm for eval progress ([`69d98ac`](https://github.com/chanind/frame-semantic-transformer/commit/69d98acb74b20e4b45571910c55f09ff8ce9c37a))

* tweaking evaluate ([`0f3db83`](https://github.com/chanind/frame-semantic-transformer/commit/0f3db83383f109e744905b3951cb6d8b46da99e0))

* adding evaluate / predict helpers ([`be08ec1`](https://github.com/chanind/frame-semantic-transformer/commit/be08ec19e9bc4638f379030621e818235f58354d))

* adding fulltext filenames from sesame for eval ([`081533b`](https://github.com/chanind/frame-semantic-transformer/commit/081533b78e3f42e0f899b995ffa9d7fc0ac1ed3e))

* removing validation loop end as well ([`1c5dfb2`](https://github.com/chanind/frame-semantic-transformer/commit/1c5dfb252bf7c6eb15eb839702dc75839abd2f35))

* removing return from training loop end ([`ace9621`](https://github.com/chanind/frame-semantic-transformer/commit/ace96214b5ad10a174d1a080b1a4a331050f86dc))

* adding in closure... ([`4b8e184`](https://github.com/chanind/frame-semantic-transformer/commit/4b8e18441b90c905b93637a9fab4a883784b4fb0))

* updating optimzier_step ([`ea79146`](https://github.com/chanind/frame-semantic-transformer/commit/ea791461efbbadc49e6abaef71b854a64b14e15b))

* fixing typo ([`7aa0a49`](https://github.com/chanind/frame-semantic-transformer/commit/7aa0a49ec4d170790be996a4fe4488e4bd2435fa))

* moving dataset generation out of the tuner ([`ac9f90f`](https://github.com/chanind/frame-semantic-transformer/commit/ac9f90f6be66e80ca15e24b6599716aaa3834378))

* adding future annotations stuff ([`611a1a5`](https://github.com/chanind/frame-semantic-transformer/commit/611a1a5eefc3b57b801e8f17f55b2ebe70d185ca))

* adding future annotations stuff ([`b6c8a53`](https://github.com/chanind/frame-semantic-transformer/commit/b6c8a53ca426badfb30bed495aea8c0954636cbc))

* setting up a model for training ([`abfc3d8`](https://github.com/chanind/frame-semantic-transformer/commit/abfc3d8a448675b2d6d1b752bbb30560dc26bdfd))

* skipping confusing frames for now ([`1888b8c`](https://github.com/chanind/frame-semantic-transformer/commit/1888b8cd5e9af73f70667cb37dc38d19e868ecb5))

* adding helper for parsing examples from docs ([`8856b2e`](https://github.com/chanind/frame-semantic-transformer/commit/8856b2edacf96c83c466baf9da9577e2e435a1a8))

* fixing mypy ([`14e4832`](https://github.com/chanind/frame-semantic-transformer/commit/14e4832b6b30e674369edbdd43c12634b8cce489))

* fixing black formatting ([`a24193a`](https://github.com/chanind/frame-semantic-transformer/commit/a24193ada5dba6d59cf5133840488184d87e3274))

* initial commit ([`4df6628`](https://github.com/chanind/frame-semantic-transformer/commit/4df6628f14781b1da9b2b9defecd27b1e37250f5))
