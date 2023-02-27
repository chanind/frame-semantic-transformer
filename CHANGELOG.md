# Changelog

<!--next-version-placeholder-->

## v0.6.1 (2023-02-27)
### Fix
* Fixing errors when no frames are found ([#14](https://github.com/chanind/frame-semantic-transformer/issues/14)) ([`ef2424c`](https://github.com/chanind/frame-semantic-transformer/commit/ef2424cba57515b9aa8205fe29f015b57f26504e))

## v0.6.0 (2023-02-23)
### Feature
* Adding support for running inference on multiple sentences in batches ([#11](https://github.com/chanind/frame-semantic-transformer/issues/11)) ([`e6423e5`](https://github.com/chanind/frame-semantic-transformer/commit/e6423e594de22850d9ea6369f80ca9a0e81ac8e0))

## v0.5.0 (2023-02-22)
### Feature
* Multilingual training refactor ([#10](https://github.com/chanind/frame-semantic-transformer/issues/10)) ([`7bf7ae5`](https://github.com/chanind/frame-semantic-transformer/commit/7bf7ae5474cc586739f1ad77848e45db09ff5a0d))

## v0.4.1 (2022-05-25)
### Fix
* Updating README stats ([`76e4e75`](https://github.com/chanind/frame-semantic-transformer/commit/76e4e754b534389de9c86caced762461ae884e88))

## v0.4.0 (2022-05-24)
### Feature
* Frame classification hints ([#3](https://github.com/chanind/frame-semantic-transformer/issues/3)) ([`201ed51`](https://github.com/chanind/frame-semantic-transformer/commit/201ed517c7a78b32d659150dadf11fcd7b4fea6d))

## v0.3.3 (2022-05-22)
### Fix
* Make trimmed batch contiguous ([#2](https://github.com/chanind/frame-semantic-transformer/issues/2)) ([`21aee70`](https://github.com/chanind/frame-semantic-transformer/commit/21aee707b25a96139f19c84c176fe692323e6b48))

## v0.3.2 (2022-05-22)
### Fix
* Add torch.no_grad() to batch trimming step ([`8b8a401`](https://github.com/chanind/frame-semantic-transformer/commit/8b8a401b339b674abeca300408d613472f6fa197))

## v0.3.1 (2022-05-22)
### Fix
* Adding LICENSE into pypi description ([`c6e0a42`](https://github.com/chanind/frame-semantic-transformer/commit/c6e0a42fc606cc06399f67966338ebbd8bc0e5d9))
* Adding README into pypi description ([`1b99551`](https://github.com/chanind/frame-semantic-transformer/commit/1b9955136ff6f4679aa093fd7a05765c035f03f4))

## v0.3.0 (2022-05-22)
### Feature
* Adding a helper to trim unnecessary padding chars for faster training / generation ([#1](https://github.com/chanind/frame-semantic-transformer/issues/1)) ([`58e58a8`](https://github.com/chanind/frame-semantic-transformer/commit/58e58a8ee19da5a57b24d6aa8ab0a440fc9e1c93))

## v0.2.1 (2022-05-22)
### Fix
* Reverting to older lock file for mypy ([`08f0c63`](https://github.com/chanind/frame-semantic-transformer/commit/08f0c638f02d7b2dfd0589b8ace2232dfd3e9fb8))
* Relaxing transformers version req ([`57464a1`](https://github.com/chanind/frame-semantic-transformer/commit/57464a19cb69a21f91a1477ec44cc5ee09a29be2))

## v0.2.0 (2022-05-21)
### Feature
* Autopublish ([`a0900ff`](https://github.com/chanind/frame-semantic-transformer/commit/a0900ff546e99be008491b9b2446d9534e08af76))

### Fix
* Pinning old version of semantic-release plugin ([`85f3a62`](https://github.com/chanind/frame-semantic-transformer/commit/85f3a625225da5c880b929c147bcf65bd629f2ea))
* Adding fetch-depth 0 for release ([`6dab4e6`](https://github.com/chanind/frame-semantic-transformer/commit/6dab4e661059cc8e895161cc07e3bec84719f784))
* Autopublish to pypi ([`3591c27`](https://github.com/chanind/frame-semantic-transformer/commit/3591c27a78ff7a687d378649b806dfacce00d775))
