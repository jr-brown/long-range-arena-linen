import System.Environment (getArgs)
import Data.Maybe (fromMaybe)
import qualified Data.Map as M
import qualified Data.List as L


--Useful type annotations
type Option = Char
type Arg = String
type Suffix = String
type OptionData = ([(Option, Arg)], [Option], [Arg])
type NewLineFlag = Bool

data Config = Config [FilePath] (String -> String) NewLineFlag OutputMode

data OutputMode = Modify (Maybe Suffix)
                | Print



noArgError :: Char -> String
noArgError x = "Option \'" ++ [x] ++ "\' required an argument but none was given"

tooManyReqArgOptionsError :: String -> String
tooManyReqArgOptionsError xs = "Options \'" ++ xs ++ "\' were all given at once but all require an argument"

-- Get all possible options with their own args from the args
-- Takes a list of all options which require an arg
-- Returns a list of any req_arg_options with their args, other options found
-- and all leftover args
getOptionData :: String -> [String] -> Either String OptionData
getOptionData req_arg_options args = getOptionData' args ([], [], [])
  where
    getOptionData' :: [String] -> OptionData -> Either String OptionData
    getOptionData' [] result = Right result

    getOptionData' [x] (aos, os, as)
      | head x == '-'  = case tail x `L.intersect` req_arg_options of
                           ""  -> Right (aos, os ++ tail x, as)
                           [y] -> Left $ noArgError y
                           ys  -> Left $ tooManyReqArgOptionsError ys
      | otherwise   = Right (aos, os, as++[x])

    getOptionData' (x1:x2:xss) (aos, os, as)
      | head x1 == '-' = case tail x1 `L.intersect` req_arg_options of
                 ""  -> getOptionData' (x2:xss) (aos, os ++ tail x1, as)
                 [y] -> case head x2 of
                          '-' -> Left $ noArgError y
                          _   -> if y `elem` map fst aos
                                    then Left $ "Option \'" ++ [y] ++ "\' was given more than once"
                                    else getOptionData' xss (aos ++ [(y, x2)], os ++ L.delete y (tail x1), as)
                 ys  -> Left $ tooManyReqArgOptionsError ys
      | otherwise      = getOptionData' (x2:xss) (aos, os, as ++ [x1])


readArgs :: [String] -> Either String Config
readArgs args = do
  (aos, os, as) <- getOptionData "ms" args
  let aoms = M.fromList aos
  m <- maybeToRight "No mode given use -m" (M.lookup 'm' aoms)
  f <- m `getFromMap` strFuncMap
  let pcfg = Config as f ('n' `elem` os)
  case (M.lookup 's' aoms, 'p' `elem` os) of
    (Just _, True)  -> Left "Can't give -p and -s"
    (Nothing, True) -> Right $ pcfg Print
    (m_sfx, False)  -> Right $ pcfg (Modify m_sfx)


splitHeadRecombine :: (Monoid a) => [a] -> (a, a)
splitHeadRecombine xs = (head xs, mconcat $ tail xs)


splitOn :: (Eq a) => a -> [a] -> [[a]]
splitOn p x = case dropWhile (==p) x of
        [] -> []
        x' -> w : splitOn p x''
          where (w, x'') = break (==p) x'


maybeToRight :: a -> Maybe b -> Either a b
maybeToRight x Nothing  = Left x
maybeToRight x (Just y) = Right y


mkListOpSafe :: ([a] -> a) -> [a] -> Maybe a
mkListOpSafe f [] = Nothing
mkListOpSafe f xs = Just $ f xs

safeLast = mkListOpSafe last
safeHead = mkListOpSafe head


catchNoInfo :: Maybe String -> String
catchNoInfo = fromMaybe "~~~ No information found ~~~"


extractFinalAccurracy :: String -> String
extractFinalAccurracy = catchNoInfo . safeLast . filter_acc . lines
  where 
    filter_acc_lines txt_lines = filter (\l -> "acc:" `L.isInfixOf` l) txt_lines
    filter_acc_words txt_words = take 2 $ dropWhile (/= "acc:") txt_words
    filter_acc = map (unwords . filter_acc_words . words) . filter_acc_lines


strFuncMap :: M.Map String (String -> String)
strFuncMap = M.fromList [ ("id", id)
                        , ("extractAcc", extractFinalAccurracy) 
                        , ("lastLine", catchNoInfo . safeLast . lines) ]

getFromMap :: String -> M.Map String a -> Either String a
getFromMap x m = maybeToRight ("\'" ++ x ++ "\' was not found") (M.lookup x m)


modifyFile :: (String -> String) -> Maybe Suffix -> FilePath -> IO ()
modifyFile func m_sfx file_name = do
  file_txt <- readFile file_name
  let new_file_txt = func file_txt
  let file_split = splitOn '.' file_name
  let (file_root, extension) = splitHeadRecombine file_split
  case m_sfx of
    Just sfx -> writeFile (file_root ++ sfx ++ "." ++ extension) new_file_txt
    Nothing  -> writeFile file_name new_file_txt


getFileOutput :: (String -> String) -> FilePath -> IO ()
getFileOutput func file_name = readFile file_name >>= putStrLn . func


main :: IO ()
main = do
  args <- getArgs
  let e_cfg = readArgs args

  case e_cfg of
    Left err  -> putStrLn $ "Error: " ++ err
    Right (Config fpaths f nl_flag out_mode) ->

      case out_mode of
        Modify m_sfx -> mapM_ (modifyFile f m_sfx) fpaths
        Print -> mconcat $ map process_file fpaths
          where
            process_file p = putStrLn ("### " ++ p ++ " ###") <> getFileOutput f p <> putStr nl
            nl = if nl_flag then "\n" else ""

