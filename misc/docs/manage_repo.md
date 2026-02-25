# Managing the Zero2Neuro Repo


We are maintaining two branches:
1. main: this is the development branch.  Make sure that you are making changes only to this branch
2. release: this is the (semi) frozen version of the repo for consumption by users.  This is also the default when someone clones the repo

## Managing your current branch

Query git as to which branch you have on your local disk:

```
git branch --show-current
```

Change to a new branch (main in this case):
```
git switch main
git pull
```

## Checking in changes

Changes should only be pushed out to the main branch

Stage files to commit:
```
git add <FILE>
git add -u      # Add all changed files
```

Then commit:
```
git commit -m "<MEANINGFUL MESSAGE>"
```

Then push:
```
git push
```

___
## Conflicted Push

If git reports that you are in conflict with the current state of the branch, the cleanest thing to do is:
```
git pull --rebase
```

This will note all changes you have made since your last pull, then pull down the latest version in the repository, and finally _replay_ all of your changes into the new state of the branch.

It is possible that there will still be conflicts left in the files.  If this is the case, then git will show both alternative states of the altered segments of the files.  For example:
```
<<<<<<< HEAD
<div class="header">
<h1>Sample text 1</h1>
</div>
=======
<div class="header">
<h1>Sample text 2</h1>
</div>
>>>>>>> feature-branch
```

### Resolve the conflict

Edit the conflicted files:
- Choose which lines to keep (you can also blend the changes)
- Remove the <<<<, ==== and >>>> lines

Tell git that you have updated the files using:

```
git add ...
``` 

Check the repository status:
```
git status
```

Then, clear the conflicts:
```
git rebase --continue
```

If there are still conflicts, then repeat.

If there are no conflicts, then push the updates:

```
git push
```

___
## New Release

1. Update the Version in src/zero2neuro.py
2. Update the version in src/CHANGELOG.md
3. Push these changes:
```
git add -u
git commit -m 'Release vX.Y.Z'
git push
```

4. Create a new tag:
```
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

5. Update release in GitHub: Releases → Draft new release → choose tag vX.Y.Z → Publish

6. Update the default version:
```
git fetch --tags
git switch --detach vX.Y.Z
git switch -C release
git push -f origin release
```

7. Switch back to main:
```
git switch main
```
___