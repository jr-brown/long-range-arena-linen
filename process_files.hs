import System.Environment (getArgs)
import qualified Data.Map as M
import qualified Data.List as L


extractAccurracy :: String -> String
extractAccurracy txt = unlines new_lines
    where new_lines = filter (\line -> "acc" `L.isInfixOf` line) (lines txt)


toUnit :: a -> ()
toUnit _ = ()


splitHeadRecombine :: (Monoid a) => [a] -> (a, a)
splitHeadRecombine xs = (head xs, mconcat $ tail xs)


splitOn :: (Eq a) => a -> [a] -> [[a]]
splitOn p x = case dropWhile (==p) x of
                [] -> []
                x' -> w : splitOn p x''
                    where (w, x'') = break (==p) x'


processFile :: (String -> String) -> String -> IO ()
processFile func file_name = do
    file_txt <- readFile file_name
    let new_file_txt = func file_txt
    let file_split = splitOn '.' file_name
    let (file_root, suffix) = splitHeadRecombine file_split
    writeFile (file_root ++ "_processed." ++ suffix) new_file_txt


modeMap :: M.Map String (String -> String)
modeMap = M.fromList [ ("id", id)
                     , ("extractAcc", extractAccurracy) ]


main :: IO ()
main = do
    args <- getArgs
    case args of
        [] -> putStrLn "No mode given"
        mode : file_names -> case M.lookup mode modeMap of
                                 Just f -> fmap toUnit (sequence file_ops)
                                     where file_ops = map (processFile f) file_names
                                 Nothing -> putStrLn "Mode not recognised"

